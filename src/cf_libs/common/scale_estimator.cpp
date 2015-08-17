#include "scale_estimator.hpp"

using namespace cf_tracking;

ScaleEstimatorParas::ScaleEstimatorParas() :
    scaleCellSize(4),
    scaleModelMaxArea(512),
    scaleStep(1.02),
    numberOfScales(33),
    scaleSigmaFactor(1.0 / 4.0),
    lambda(0.01),
    learningRate(0.025),
    useFhogTranspose(false),
    resizeType(cv::INTER_LINEAR),
    originalVersion(false)
{};

ScaleEstimator::ScaleEstimator(ScaleEstimatorParas paras) :
    _frameIdx(0),
    _isInitialized(false),
    _scaleModelFactor(0),
    fhogToCvCol(0),
    _MIN_SCALE_FACTOR(0.01),
    _MAX_SCALE_FACTOR(40),
    _SCALE_CELL_SIZE(paras.scaleCellSize),
    _SCALE_MODEL_MAX_AREA(paras.scaleModelMaxArea),
    _SCALE_STEP(paras.scaleStep),
    _N_SCALES(paras.numberOfScales),
    _SCALE_SIGMA_FACTOR(paras.scaleSigmaFactor),
    _LAMBDA(paras.lambda),
    _LEARNING_RATE(paras.learningRate),
    _TYPE(cv::DataType<float>::type),
    _RESIZE_TYPE(paras.resizeType),
    _ORIGINAL_VERSION(paras.originalVersion) {
    // init dft
    cv::Mat initDft = (cv::Mat_<float>(1, 1) << 1);
    dft(initDft, initDft);

    if (paras.useFhogTranspose)
    { fhogToCvCol = &piotr::fhogToCvColT; }
    else
    { fhogToCvCol = &piotr::fhogToCol; }
}

ScaleEstimator::~ScaleEstimator() {}

bool ScaleEstimator::reinit(const cv::Mat& image, const Point& pos,
                            const Size& targetSize, const float& currentScaleFactor) {
    _targetSize = targetSize;
    // scale filter output target
    float scaleSigma = (sqrt(_N_SCALES) * _SCALE_SIGMA_FACTOR);
    cv::Mat colScales = numberToColVector<float>(_N_SCALES);
    float scaleHalf = (ceil(_N_SCALES / 2.0));

    cv::Mat ss = colScales - scaleHalf;
    cv::Mat ys;
    exp(-0.5 * ss.mul(ss) / (scaleSigma * scaleSigma), ys);

    cv::Mat ysf;
    // always use CCS here; regular COMPLEX_OUTPUT is bugged
    cv::dft(ys, ysf, cv::DFT_ROWS);

    // scale filter cos window
    if (_N_SCALES % 2 == 0) {
        _scaleWindow = hanningWindow<float>(_N_SCALES + 1);
        _scaleWindow = _scaleWindow.rowRange(1, _scaleWindow.rows);
    } else {
        _scaleWindow = hanningWindow<float>(_N_SCALES);
    }

    ss = scaleHalf - colScales;
    _scaleFactors = pow<float, float>(_SCALE_STEP, ss);
    _scaleModelFactor = sqrt(_SCALE_MODEL_MAX_AREA / targetSize.area());
    _scaleModelSz = sizeFloor(targetSize *  _scaleModelFactor);

    // expand ysf to have the number of rows of scale samples
    int ysfRow = static_cast<int>(floor(_scaleModelSz.width / _SCALE_CELL_SIZE)
                                  * floor(_scaleModelSz.height / _SCALE_CELL_SIZE) * FFC::numberOfChannels());

    _ysf = repeat(ysf, ysfRow, 1);

    cv::Mat sfNum, sfDen;

    if (getScaleTrainingData(image, pos,
                             currentScaleFactor, sfNum, sfDen) == false)
    { return false; }

    _sfNumerator = sfNum;
    _sfDenominator = sfDen;

    _isInitialized = true;
    ++_frameIdx;
    return true;
}

bool ScaleEstimator::detectScale(const cv::Mat& image, const Point& pos,
                                 float& currentScaleFactor) const {
    cv::Mat xs;

    if (getScaleFeatures(image, pos, xs, currentScaleFactor) == false)
    { return false; }

    cv::Mat xsf;
    dft(xs, xsf, cv::DFT_ROWS);

    mulSpectrums(_sfNumerator, xsf, xsf, cv::DFT_ROWS);
    reduce(xsf, xsf, 0, cv::REDUCE_SUM, -1);

    cv::Mat sfDenLambda;
    sfDenLambda = addRealToSpectrum<float>(_LAMBDA, _sfDenominator, cv::DFT_ROWS);

    cv::Mat responseSf;
    divSpectrums(xsf, sfDenLambda, responseSf, cv::DFT_ROWS, false);

    cv::Mat scaleResponse;
    idft(responseSf, scaleResponse, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE | cv::DFT_ROWS);

    cv::Point recoveredScale;
    double maxScaleResponse;
    minMaxLoc(scaleResponse, 0, &maxScaleResponse, 0, &recoveredScale);

    currentScaleFactor *= _scaleFactors.at<float>(recoveredScale);

    currentScaleFactor = std::max(currentScaleFactor, _MIN_SCALE_FACTOR);
    currentScaleFactor = std::min(currentScaleFactor, _MAX_SCALE_FACTOR);
    return true;
}

bool ScaleEstimator::updateScale(const cv::Mat& image, const Point& pos,
                                 const float& currentScaleFactor) {
    ++_frameIdx;
    cv::Mat sfNum, sfDen;

    if (getScaleTrainingData(image, pos, currentScaleFactor,
                             sfNum, sfDen) == false)
    { return false; }

    // both summands are in CCS packaged format; thus adding is OK
    _sfDenominator = (1 - _LEARNING_RATE) * _sfDenominator + _LEARNING_RATE * sfDen;
    _sfNumerator = (1 - _LEARNING_RATE) * _sfNumerator + _LEARNING_RATE * sfNum;
    return true;
}

bool ScaleEstimator::getScaleTrainingData(const cv::Mat& image,
        const Point& pos,
        const float& currentScaleFactor,
        cv::Mat& sfNum, cv::Mat& sfDen) const {
    cv::Mat xs;

    if (getScaleFeatures(image, pos, xs, currentScaleFactor) == false)
    { return false; }

    cv::Mat xsf;
    dft(xs, xsf, cv::DFT_ROWS);
    mulSpectrums(_ysf, xsf, sfNum, cv::DFT_ROWS, true);
    cv::Mat mulTemp;
    mulSpectrums(xsf, xsf, mulTemp, cv::DFT_ROWS, true);
    reduce(mulTemp, sfDen, 0, cv::REDUCE_SUM, -1);
    return true;
}

bool ScaleEstimator::getScaleFeatures(const cv::Mat& image, const Point& pos,
                                      cv::Mat& features, float scale) const {
    int colElems = _ysf.rows;
    features = cv::Mat::zeros(colElems, _N_SCALES, _TYPE);
    cv::Mat patch;
    cv::Mat patchResized;
    cv::Mat patchResizedFloat;
    cv::Mat firstPatch;
    float cosFactor = -1;

    // do not extract features for first and last scale,
    // since the scaleWindow will always multiply these with 0;
    // extract first required sub window separately; smaller scales are extracted
    // from this patch to avoid multiple border replicates on out of image patches
    int idxScale = 1;
    float patchScale = scale * _scaleFactors.at<float>(0, idxScale);
    Size firstPatchSize = sizeFloor(_targetSize * patchScale);
    Point posInFirstPatch(0, 0);
    cosFactor = _scaleWindow.at<float>(idxScale, 0);

    if (getSubWindow(image, firstPatch, firstPatchSize, pos, &posInFirstPatch) == false)
    { return false; }

    if (_ORIGINAL_VERSION)
    { depResize(firstPatch, patchResized, _scaleModelSz); }
    else
    { cv::resize(firstPatch, patchResized, _scaleModelSz, 0, 0, _RESIZE_TYPE); }

    patchResized.convertTo(patchResizedFloat, CV_32FC(3));
    fhogToCvCol(patchResizedFloat, features, _SCALE_CELL_SIZE, idxScale, cosFactor);

    for (idxScale = 2; idxScale < _N_SCALES - 1; ++idxScale) {
        float patchScale = scale * _scaleFactors.at<float>(0, idxScale);
        Size patchSize = sizeFloor(_targetSize * patchScale);
        cosFactor = _scaleWindow.at<float>(idxScale, 0);

        if (getSubWindow(firstPatch, patch, patchSize, posInFirstPatch) == false)
        { return false; }

        if (_ORIGINAL_VERSION)
        { depResize(patch, patchResized, _scaleModelSz); }
        else
        { cv::resize(patch, patchResized, _scaleModelSz, 0, 0, _RESIZE_TYPE); }

        patchResized.convertTo(patchResizedFloat, CV_32FC(3));
        fhogToCvCol(patchResizedFloat, features, _SCALE_CELL_SIZE, idxScale, cosFactor);
    }

    return true;
}

