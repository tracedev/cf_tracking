/* This class represents a C++ implementation of the Discriminative Scale
Space Tracker (DSST) [1]. The class contains the 2D translational filter.
The 1D scale filter can be found in scale_estimator.hpp.

It is implemented closely to the Matlab implementation by the original authors:
http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html
However, some implementation details differ and some difference in performance
has to be expected.

Additionally, target loss detection is implemented according to [2].

Every complex matrix is as default in CCS packed form:
see: https://software.intel.com/en-us/node/504243
and http://docs.opencv.org/modules/core/doc/operations_on_arrays.html

References:
[1] M. Danelljan, et al.,
"Accurate Scale Estimation for Robust Visual Tracking,"
in Proc. BMVC, 2014.

[2] D. Bolme, et al.,
�Visual Object Tracking using Adaptive Correlation Filters,�
in Proc. CVPR, 2010.
*/

#ifndef DSST_TRACKER_HPP_
#define DSST_TRACKER_HPP_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/traits.hpp>
#include <memory>
#include <iostream>
#include <fstream>

#include "cv_ext.hpp"
#include "mat_consts.hpp"
#include "feature_channels.hpp"
#include "gradientMex.hpp"
#include "math_helper.hpp"
#include "cf_tracker.hpp"
#include "scale_estimator.hpp"
#include "psr.hpp"

namespace cf_tracking {
    struct DsstParameters {
        DsstParameters();

        double padding;
        double outputSigmaFactor;
        double lambda;
        double learningRate;
        int templateSize;
        int cellSize;

        bool enableTrackingLossDetection;
        double psrThreshold;
        int psrPeakDel;

        bool enableScaleEstimator;
        double scaleSigmaFactor;
        double scaleStep;
        int scaleCellSize;
        int numberOfScales;

        //testing
        bool originalVersion;
        int resizeType;
        bool useFhogTranspose;
    };

    class DsstTracker : public CfTracker {
    public:
        typedef float T; // set precision here double or float
        static const int CV_TYPE = cv::DataType<T>::type;
        typedef cv::Size_<T> Size;
        typedef cv::Point_<T> Point;
        typedef cv::Rect_<T> Rect;
        typedef typename FhogFeatureChannels<T>::type FFC;
        typedef typename DsstFeatureChannels<T>::type DFC;
        typedef mat_consts::constants<T> consts;

        DsstTracker(DsstParameters paras);
        virtual ~DsstTracker();

        bool reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox);

        virtual void set_learning_rate(double learningRate);
        virtual void set_padding(double padding);
        virtual void set_lambda( double lambda);
        virtual void set_output_sigma_factor(double outputSigmaFactor);
        virtual void set_psr_threshold(double psrThreshold);
        virtual void set_cell_size(int cellSize);
        virtual void set_template_size(int templateSize);
        virtual bool update(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual bool updateAt(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual const std::string getId();
        virtual  bool get_found();

    private:
        bool reinit_(const cv::Mat& image, Rect& boundingBox);
        bool getTranslationTrainingData(const cv::Mat& image, std::shared_ptr<DFC>& hfNum, cv::Mat& hfDen, const Point& pos) const;
        bool getTranslationFeatures(const cv::Mat& image, std::shared_ptr<DFC>& features, const Point& pos, T scale) const;
        bool update_(const cv::Mat& image, Rect& boundingBox);
        bool updateAt_(const cv::Mat& image, Rect& boundingBox);
        bool updateAtScalePos(const cv::Mat& image, const Point& oldPos, const T oldScale, Rect& boundingBox);
        bool evalReponse(const cv::Mat& image, const cv::Mat& response, const cv::Point2i& maxResponseIdx, const Rect& tempBoundingBox) const;
        bool detectModel(const cv::Mat& image, cv::Mat& response, cv::Point2i& maxResponseIdx, Point& newPos, T& newScale) const;
        bool updateModel(const cv::Mat& image, const Point& newPos, T newScale);

        typedef void(*cvFhogPtr)
        (const cv::Mat& img, std::shared_ptr<DFC>& cvFeatures, int binSize, int fhogChannelsToCopy);
        cvFhogPtr cvFhog;

        typedef void(*dftPtr)
        (const cv::Mat& input, cv::Mat& output, int flags);
        dftPtr calcDft;

        cv::Mat _cosWindow;
        cv::Mat _y;
        std::shared_ptr<DFC> _hfNumerator;
        cv::Mat _hfDenominator;
        cv::Mat _yf;
        Point _pos;
        Size _templateSz;
        Size _templateSizeNoFloor;
        Size _baseTargetSz;
        Rect _lastBoundingBox;
        T _scale; // _scale is the scale of the template; not the target
        T _templateScaleFactor; // _templateScaleFactor is used to calc the target scale
        ScaleEstimator* _scaleEstimator;
        int _frameIdx;
        bool _isInitialized;

        double _MIN_AREA;
        double _MAX_AREA_FACTOR;
        T _PADDING;
        T _OUTPUT_SIGMA_FACTOR;
        T _LAMBDA;
        T _LEARNING_RATE;
        T _PSR_THRESHOLD;
        int _PSR_PEAK_DEL;
        int _CELL_SIZE;
        int _TEMPLATE_SIZE;
        std::string _ID;
        bool _ENABLE_TRACKING_LOSS_DETECTION;
        int _RESIZE_TYPE;
        bool _ORIGINAL_VERSION;
        bool _USE_CCS;
        bool _FOUND;
    };
}

#endif /* KCF_TRACKER_H_ */
