/* This class represents the 1D correlation filter proposed in  [1]. It is used to estimate the
scale of a target.

It is implemented closely to the Matlab implementation by the original authors:
http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html
However, some implementation details differ and some difference in performance
has to be expected.

Every complex matrix is as default in CCS packed form:
see : https://software.intel.com/en-us/node/504243
and http://docs.opencv.org/modules/core/doc/operations_on_arrays.html

References:
[1] M. Danelljan, et al.,
"Accurate Scale Estimation for Robust Visual Tracking,"
in Proc. BMVC, 2014.

*/

#ifndef SCALE_ESTIMATOR_HPP_
#define SCALE_ESTIMATOR_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/core/traits.hpp>
#include <algorithm>

#include "mat_consts.hpp"
#include "cv_ext.hpp"
#include "feature_channels.hpp"
#include "gradientMex.hpp"
#include "math_helper.hpp"

namespace cf_tracking {
    struct ScaleEstimatorParas {
        ScaleEstimatorParas();
        int scaleCellSize;
        float scaleModelMaxArea;
        float scaleStep;
        int numberOfScales;
        float scaleSigmaFactor;

        float lambda;
        float learningRate;

        // testing
        bool useFhogTranspose;
        int resizeType;
        bool originalVersion;
    };

    class ScaleEstimator {
    public:
        typedef typename FhogFeatureChannels<float>::type FFC;
        typedef cv::Size_<float> Size;
        typedef cv::Point_<float> Point;
        typedef mat_consts::constants<float> consts;

        ScaleEstimator(ScaleEstimatorParas paras);
        virtual ~ScaleEstimator();

        bool reinit(const cv::Mat& image, const Point& pos,
                    const Size& targetSize, const float& currentScaleFactor);

        bool detectScale(const cv::Mat& image, const Point& pos,
                         float& currentScaleFactor) const;
        bool updateScale(const cv::Mat& image, const Point& pos,
                         const float& currentScaleFactor);
    private:
        bool getScaleTrainingData(const cv::Mat& image,
                                  const Point& pos,
                                  const float& currentScaleFactor,
                                  cv::Mat& sfNum, cv::Mat& sfDen) const;

        bool getScaleFeatures(const cv::Mat& image, const Point& pos,
                              cv::Mat& features, float scale) const;

        typedef void(*fhogToCvRowPtr)
        (const cv::Mat& img, cv::Mat& cvFeatures, int binSize, int rowIdx, float cosFactor);
        fhogToCvRowPtr fhogToCvCol;

        cv::Mat _scaleWindow;
        float _scaleModelFactor;
        cv::Mat _sfNumerator;
        cv::Mat _sfDenominator;
        cv::Mat _scaleFactors;
        Size _scaleModelSz;
        Size _targetSize;
        cv::Mat _ysf;
        int _frameIdx;
        bool _isInitialized;

        const int _TYPE;
        const int _SCALE_CELL_SIZE;
        const float _SCALE_MODEL_MAX_AREA;
        const float _SCALE_STEP;
        const int _N_SCALES;
        const float _SCALE_SIGMA_FACTOR;
        const float _LAMBDA;
        const float _LEARNING_RATE;
        const int _RESIZE_TYPE;
        // it should be possible to find more reasonable values for min/max scale; application dependent
        float _MIN_SCALE_FACTOR;
        float _MAX_SCALE_FACTOR;

        const bool _ORIGINAL_VERSION;
    };
}

#endif
