#ifndef TRACKER_HPP_
#define TRACKER_HPP_

#include "opencv2/core/core.hpp"

namespace cf_tracking {
    class CfTracker {
    public:
        virtual ~CfTracker() {};

        virtual bool update(const cv::Mat& image, cv::Rect_<int>& boundingBox) = 0;
        virtual bool reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox) = 0;
        virtual bool updateAt(const cv::Mat& image, cv::Rect_<int>& boundingBox) = 0;

        virtual const std::string getId() = 0;
    };
}
#endif
