
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <set>
#include <unordered_map>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev,
        std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    ///@brief fill matches belonging to a shrinked bounding boxes in current
    /// frame. Then calculate Mu & Sigma, then discard outliers > sigma
    /// (outside 68% confidence interval)
    cv::Rect curr_shrinked_box{};
    std::vector<cv::DMatch> curr_shrinked_box_matches{};
    float shrink_factor{0.15f};

    curr_shrinked_box.x = static_cast<int>(boundingBox.roi.x + shrink_factor * boundingBox.roi.width / 2.0);
    curr_shrinked_box.y = static_cast<int>(boundingBox.roi.y + shrink_factor * boundingBox.roi.height / 2.0);
    curr_shrinked_box.width = static_cast<int>(boundingBox.roi.width * (1 - shrink_factor));
    curr_shrinked_box.height = static_cast<int>(boundingBox.roi.height * (1 - shrink_factor));

    for (const auto& match : kptMatches)
    {
        auto& curr_point{kptsCurr[match.trainIdx]};
        if (curr_shrinked_box.contains(curr_point.pt))
        {
            curr_shrinked_box_matches.emplace_back(match);
        }
    }
    if (curr_shrinked_box_matches.empty()) return;
    ///@brief compute mean & std of matched distance between individual keypoints -- not whole describtor
    auto mean{std::accumulate(curr_shrinked_box_matches.begin(), curr_shrinked_box_matches.end(),
            0.0f, [&kptsCurr, &kptsPrev](const float a, const cv::DMatch b)
            { return a + cv::norm(kptsCurr[b.trainIdx].pt - kptsPrev[b.queryIdx].pt);}) / curr_shrinked_box_matches.size()};

    std::vector<float> diff (curr_shrinked_box_matches.size(), 0.0f);

    std::transform(curr_shrinked_box_matches.begin(), curr_shrinked_box_matches.end(), diff.begin(),
            [&mean, &kptsCurr, &kptsPrev](const cv::DMatch& b)
            { return cv::norm(kptsCurr[b.trainIdx].pt - kptsPrev[b.queryIdx].pt) - mean;});
    auto squared_sum{std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0f)};
    auto std{std::sqrt(squared_sum / diff.size())};

    float upper_bound{mean + std};
    float lower_bound{mean - std};

    for (const auto& match : curr_shrinked_box_matches)
    {
        auto dis{cv::norm(kptsCurr[match.trainIdx].pt - kptsPrev[match.queryIdx].pt)};

        if ((lower_bound < dis) and (dis < upper_bound))
        {
            boundingBox.kptMatches.emplace_back(match);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // Compute distance ratios on every pair of keypoints, O(n^2) on the number of matches contained within the ROI
    vector<double> distRatios;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);  // kptsCurr is indexed by trainIdx, see NOTE in matchBoundinBoxes
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);  // kptsPrev is indexed by queryIdx, see NOTE in matchBoundinBoxes

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);  // kptsCurr is indexed by trainIdx, see NOTE in matchBoundinBoxes
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);  // kptsPrev is indexed by queryIdx, see NOTE in matchBoundinBoxes

            // Use cv::norm to calculate the current and previous Euclidean distances between each keypoint in the pair
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            double minDist = 100.0;  // Threshold the calculated distRatios by requiring a minimum current distance between keypoints

            // Avoid division by zero and apply the threshold
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // Only continue if the vector of distRatios is not empty
    if (distRatios.size() == 0)
    {
        TTC = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // Use the median to exclude outliers
    std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = distRatios[distRatios.size() / 2];

    // Finally, calculate a TTC estimate based on these 2D camera features
    TTC = (-1.0 / frameRate) / (1 - medianDistRatio);

}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    ///@brief take the mean of the closest 50 points, sort first
    ///@todo not the best way, may be better to take the mode of values with some tolerance given
    static auto buffer{50.0f};
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), [](const LidarPoint& lhs, const LidarPoint& rhs)
    { return lhs.x < rhs.x;});
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), [](const LidarPoint& lhs, const LidarPoint& rhs)
    { return lhs.x < rhs.x;});
    auto d1 = std::accumulate(lidarPointsCurr.begin(), lidarPointsCurr.begin() + buffer, 0.0f, [](const float& f, const LidarPoint& rhs){ return f + rhs.x;}) / buffer;
    auto d0 = std::accumulate(lidarPointsPrev.begin(), lidarPointsPrev.begin() + buffer, 0.0f, [](const float& f, const LidarPoint& rhs){ return f + rhs.x;}) / buffer;

    // TTC = d1 * delta_t / (d0 - d1)
    // d0 is the previous frame's closing distance (front-to-rear bumper)
    // d1 is the current frame's closing distance (front-to-rear bumper)
    // delta_t is the time elapsed between images (1 / frameRate)
    TTC = d1 * (1.0 / frameRate) / (d0 - d1);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap<int, int> mmap{};
    set<int> currBoxIds{};

    for (auto match : matches) {

        auto prevKp{prevFrame.keypoints[match.queryIdx]};
        auto currKp{currFrame.keypoints[match.trainIdx]};

        bool prevBoxIdFound{false};
        bool currBoxIDFound{false};

        int prevBoxId{};
        int currBoxId{};

        for (const auto &bbox : prevFrame.boundingBoxes) {
            // For each bounding box in the previous frame
            if (bbox.roi.contains(prevKp.pt)) {
                prevBoxIdFound = true;
                prevBoxId = bbox.boxID;
            }
        }
        if (prevBoxIdFound) {
            // For each bounding box in the current frame
            for (const auto &bbox : currFrame.boundingBoxes) {
                if (bbox.roi.contains(currKp.pt)) {
                    currBoxIDFound = true;
                    currBoxId = bbox.boxID;
                    // Add current box ids to set
                    currBoxIds.emplace(currBoxId);
                }
            }
        }
        if (prevBoxIdFound and currBoxIDFound) {
            // Add the containing boxID for each match to a multimap
            mmap.insert({currBoxId, prevBoxId});
        }
    }
    // Loop on box_ids which are in multimap to find highest count pair
    for (const auto& box_id : currBoxIds)
    {
        // this has all box ids associated with box_id
        // [0,1], [0,2], [0,0]....
        // it will contain iterators to a range these elements
        auto all_pairs_of_current_box_id = mmap.equal_range(box_id);
        // this to count occurrences of each of element to see who has highest count
        unordered_map<int, int> counts{};
        for (auto it = all_pairs_of_current_box_id.first; it != all_pairs_of_current_box_id.second; ++it)
        {
            ++counts[it->second];
        }
        int max_occu{0};
        int matched_pair{};
        for (const auto& pair : counts)
        {
            if (pair.second > max_occu)
            {
                max_occu = pair.second;
                matched_pair = pair.first;
            }
        }
        bbBestMatches.emplace(matched_pair, box_id);
    }
}
