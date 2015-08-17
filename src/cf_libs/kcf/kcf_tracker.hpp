/* This class represents a C++ implementation of the Kernelized
Correlation Filter tracker (KCF) [1].

It is implemented closely to the Matlab implementation by the original authors:
http://home.isr.uc.pt/~henriques/circulant/
However, some implementation details differ and some difference in performance
has to be expected.

This specific implementation features the scale adaption, sub-pixel
accuracy for the correlation response evaluation and a more robust
filter update scheme [2] used by Henriques, et al. in the VOT Challenge 2014.

As default scale adaption, the tracker uses the 1D scale filter
presented in [3]. The scale filter can be found in scale_estimator.hpp.
Additionally, target loss detection is implemented according to [4].

Every complex matrix is as default in CCS packed form:
see : https://software.intel.com/en-us/node/504243
and http://docs.opencv.org/modules/core/doc/operations_on_arrays.html

References:
[1] J. Henriques, et al.,
"High-Speed Tracking with Kernelized Correlation Filters,"
PAMI, 2015.

[2] M. Danelljan, et al.,
�Adaptive Color Attributes for Real-Time Visual Tracking,�
in Proc. CVPR, 2014.

[3] M. Danelljan,
"Accurate Scale Estimation for Robust Visual Tracking,"
Proceedings of the British Machine Vision Conference BMVC, 2014.

[4] D. Bolme, et al.,
�Visual Object Tracking using Adaptive Correlation Filters,�
in Proc. CVPR, 2010.
*/

#ifndef KCF_TRACKER_HPP_
#define KCF_TRACKER_HPP_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <algorithm>

#include "cv_ext.hpp"
#include "feature_channels.hpp"
#include "gradientMex.hpp"
#include "mat_consts.hpp"
#include "math_helper.hpp"
#include "cf_tracker.hpp"
#include "scale_estimator.hpp"
#include "psr.hpp"

namespace cf_tracking {
    struct KcfParameters {
        KcfParameters();
        double padding;
        double lambda;
        double outputSigmaFactor;
        double votScaleStep;
        double votScaleWeight;
        int templateSize;
        double interpFactor;
        double kernelSigma;
        int cellSize;
        int pixelPadding;

        bool enableTrackingLossDetection;
        double psrThreshold;
        int psrPeakDel;

        bool useVotScaleEstimation;
        bool useDsstScaleEstimation;
        double scaleSigmaFactor;
        double scaleEstimatorStep;
        double scaleLambda;
        int scaleCellSize;
        int numberOfScales;

        int resizeType;
        bool useFhogTranspose;
    };
    class KcfTracker : public CfTracker {
    public:
        static const int NUM_FEATURE_CHANNELS = 31;
        typedef float T; // set precision here: double or float
        static const int CV_TYPE = cv::DataType<T>::type;
        typedef cv::Size_<T> Size;
        typedef typename FhogFeatureChannels<T>::type FFC;
        typedef mat_consts::constants<T> consts;
        typedef cv::Point_<T> Point;
        typedef cv::Rect_<T> Rect;

        KcfTracker(KcfParameters paras);
        virtual ~KcfTracker();

        virtual bool reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual void set_scale_step(double scaleStep);
        virtual void set_scale_weight(double scaleWeight);
        virtual void set_interp_factor(double interpFactor);
        virtual void set_padding(double padding);
        virtual void set_lambda( double lambda);
        virtual void set_output_sigma_factor(double outputSigmaFactor);
        virtual void set_kernel_sigma(double kernelSigma);
        virtual void set_psr_threshold(double psrThreshold);
        virtual bool update(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual bool updateAt(const cv::Mat& image, cv::Rect_<int>& boundingBox);
        virtual const std::string getId();

    private:
        bool reinit_(const cv::Mat& image, Rect& boundingBox);
        bool getTrainingData(const cv::Mat& image, cv::Mat& numeratorf, cv::Mat& denominatorf, std::shared_ptr<FFC>& xf);
        cv::Mat gaussianCorrelation(const std::shared_ptr<FFC>& xf, const std::shared_ptr<FFC>& yf) const;
        void calcGaussianTerm(cv::Mat& xy, T numel, T xx, T yy) const;
        bool getFeatures(const cv::Mat& image, const Point& pos, const T scale, std::shared_ptr<FFC>& features) const;
        bool update_(const cv::Mat& image, Rect& boundingBox);
        bool updateAt_(const cv::Mat& image, Rect& boundingBox);
        bool updateAtScalePos(const cv::Mat& image, const Point& oldPos, const T oldScale, Rect& boundingBox);
        bool evalReponse(const cv::Mat& image, const cv::Mat& response, const cv::Point2i& maxResponseIdx, const Rect& tempBoundingBox) const;
        bool detectModel(const cv::Mat& image, cv::Mat& response, cv::Point2i& maxResponseIdx, Point& newPos, T& newScale) const;
        bool updateModel(const cv::Mat& image, const Point& newPos, const T& newScale);
        bool detectScales(const cv::Mat& image, const Point& pos, cv::Mat& response, cv::Point2i& maxResponseIdx, T& scale) const;
        bool getResponse(const cv::Mat& image, const Point& pos, T scale, cv::Mat& newResponse, double& newMaxResponse, cv::Point2i& newMaxIdx) const;
        bool detect(const cv::Mat& image, const Point& pos, T scale, cv::Mat& response) const;

        typedef cv::Mat(KcfTracker::*correlatePtr)(const std::shared_ptr<FFC>&,
                const std::shared_ptr<FFC>&) const;
        correlatePtr correlate;

        typedef void(*cvFhogPtr)
        (const cv::Mat& img, std::shared_ptr<FFC>& cvFeatures, int binSize, int fhogChannelsToCopy);
        cvFhogPtr cvFhog;

        cv::Mat _cosWindow;
        cv::Mat _y;
        std::shared_ptr<FFC> _modelXf;
        cv::Mat _modelNumeratorf;
        cv::Mat _modelDenominatorf;
        cv::Mat _modelAlphaf;
        cv::Mat _yf;
        cv::Mat _scaleFactors;
        Rect _lastBoundingBox;
        Point _pos;
        Size _targetSize;
        Size _templateSz;
        T _scale;
        T _templateScaleFactor;
        int _frameIdx;
        bool _isInitialized;
        ScaleEstimator* _scaleEstimator;

        const double _MIN_AREA;
        const double _MAX_AREA_FACTOR;
        T _PADDING;
        T _LAMBDA;
        T _OUTPUT_SIGMA_FACTOR;
        T _SCALE_STEP;
        T _SCALE_WEIGHT;
        T _INTERP_FACTOR;
        T _KERNEL_SIGMA;
        T _PSR_THRESHOLD;
        const int _TEMPLATE_SIZE;
        const int _PSR_PEAK_DEL;
        const int _CELL_SIZE;
        const int _N_SCALES_VOT;
        const int _PIXEL_PADDING;
        const int _RESIZE_TYPE;
        const std::string _ID;
        const bool _USE_VOT_SCALE_ESTIMATION;
        const bool _ENABLE_TRACKING_LOSS_DETECTION;
        const bool _USE_CCS;
        // it should be possible to find more reasonable values for min/max scale; application dependent
        T _VOT_MIN_SCALE_FACTOR;
        T _VOT_MAX_SCALE_FACTOR;
    };
}

#endif /* KCF_TRACKER_H_ */
