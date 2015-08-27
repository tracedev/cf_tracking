#include "dsst_tracker.hpp"

using namespace cf_tracking;

DsstParameters::DsstParameters() :
    padding(),
    outputSigmaFactor(),
    lambda(static_cast<double>(0.01)),
    learningRate(static_cast<double>(0.012)),
    templateSize(100),
    cellSize(2),
    enableTrackingLossDetection(false),
    psrThreshold(13.5),
    psrPeakDel(1),
    enableScaleEstimator(true),
    scaleSigmaFactor(static_cast<double>(0.25)),
    scaleStep(static_cast<double>(1.02)),
    scaleCellSize(4),
    numberOfScales(33),
    originalVersion(false),
    resizeType(cv::INTER_LINEAR),
    useFhogTranspose(false)
{};

DsstTracker::DsstTracker(DsstParameters paras)
    : _isInitialized(false),
      _scaleEstimator(0),
      cvFhog(0),
      calcDft(0),
      _frameIdx(1),
      _PADDING(static_cast<float>(paras.padding)),
      _OUTPUT_SIGMA_FACTOR(static_cast<float>(paras.outputSigmaFactor)),
      _LAMBDA(static_cast<float>(paras.lambda)),
      _LEARNING_RATE(static_cast<float>(paras.learningRate)),
      _CELL_SIZE(paras.cellSize),
      _TEMPLATE_SIZE(paras.templateSize),
      _PSR_THRESHOLD(static_cast<float>(paras.psrThreshold)),
      _PSR_PEAK_DEL(paras.psrPeakDel),
      _MIN_AREA(10),
      _MAX_AREA_FACTOR(0.8),
      _ID("DSSTcpp"),
      _ORIGINAL_VERSION(paras.originalVersion),
      _RESIZE_TYPE(paras.resizeType),
      _USE_CCS(true),
      _FOUND(true) {
    if (paras.enableScaleEstimator) {
        ScaleEstimatorParas sp;
        sp.scaleCellSize = paras.scaleCellSize;
        sp.scaleStep = static_cast<float>(paras.scaleStep);
        sp.numberOfScales = paras.numberOfScales;
        sp.scaleSigmaFactor = static_cast<float>(paras.scaleSigmaFactor);
        sp.lambda = static_cast<float>(paras.lambda);
        sp.learningRate = static_cast<float>(paras.learningRate);
        sp.useFhogTranspose = paras.useFhogTranspose;
        sp.resizeType = paras.resizeType;
        sp.originalVersion = paras.originalVersion;
        _scaleEstimator = new ScaleEstimator(sp);
    }

    if (paras.useFhogTranspose)
    { cvFhog = &piotr::cvFhogT < T, DFC > ; }
    else
    { cvFhog = &piotr::cvFhog < T, DFC > ; }

    if (_USE_CCS)
    { calcDft = &cf_tracking::dftCcs; }
    else
    { calcDft = &cf_tracking::dftNoCcs; }

    // init dft
    cv::Mat initDft = (cv::Mat_<float>(1, 1) << 1);
    calcDft(initDft, initDft, 0);
}

DsstTracker::~DsstTracker() {
    delete _scaleEstimator;
}

bool DsstTracker::reinit(const cv::Mat& image, cv::Rect_<int>& boundingBox) {
    Rect bb = Rect(
                  static_cast<int>(boundingBox.x),
                  static_cast<int>(boundingBox.y),
                  static_cast<int>(boundingBox.width),
                  static_cast<int>(boundingBox.height)
              );

    return reinit_(image, bb);
}

void DsstTracker::set_learning_rate(double learningRate) {
    _LEARNING_RATE  = static_cast<float>(learningRate);
}

void DsstTracker::set_padding(double padding) {
    _PADDING  = static_cast<float>(padding);
}

void DsstTracker::set_lambda(double lambda) {
    _LAMBDA  = static_cast<float>(lambda);
}

void DsstTracker::set_output_sigma_factor(double outputSigmaFactor) {
    _OUTPUT_SIGMA_FACTOR  = static_cast<float>(outputSigmaFactor);
}

void DsstTracker::set_psr_threshold(double psrThreshold) {
    _PSR_THRESHOLD  = static_cast<float>(psrThreshold);
}

void DsstTracker::set_cell_size(int cellSize) {
    _CELL_SIZE = cellSize;
}

void DsstTracker::set_template_size(int templateSize) {
    _TEMPLATE_SIZE = templateSize;
}

bool DsstTracker::update(const cv::Mat& image, cv::Rect_<int>& boundingBox) {
    Rect bb = Rect(
                  static_cast<int>(boundingBox.x),
                  static_cast<int>(boundingBox.y),
                  static_cast<int>(boundingBox.width),
                  static_cast<int>(boundingBox.height)
              );

    if (update_(image, bb) == false)
    { return false; }

    boundingBox.x = static_cast<int>(round(bb.x));
    boundingBox.y = static_cast<int>(round(bb.y));
    boundingBox.width = static_cast<int>(round(bb.width));
    boundingBox.height = static_cast<int>(round(bb.height));

    return true;
}

bool DsstTracker::updateAt(const cv::Mat& image, cv::Rect_<int>& boundingBox) {
    bool isValid = false;

    Rect bb = Rect(
                  static_cast<int>(boundingBox.x),
                  static_cast<int>(boundingBox.y),
                  static_cast<int>(boundingBox.width),
                  static_cast<int>(boundingBox.height)
              );

    isValid = updateAt_(image, bb);

    boundingBox.x = static_cast<int>(round(bb.x));
    boundingBox.y = static_cast<int>(round(bb.y));
    boundingBox.width = static_cast<int>(round(bb.width));
    boundingBox.height = static_cast<int>(round(bb.height));

    return isValid;
}

const std::string DsstTracker::getId() {
    return _ID;
}

bool DsstTracker::reinit_(const cv::Mat& image, Rect& boundingBox) {
    _pos.x = floor(boundingBox.x) + floor(boundingBox.width * consts::c0_5);
    _pos.y = floor(boundingBox.y) + floor(boundingBox.height * consts::c0_5);
    Size targetSize = Size(boundingBox.width, boundingBox.height);

    _templateSz = Size(floor(targetSize.width * (1 + _PADDING)),
                       floor(targetSize.height * (1 + _PADDING)));

    _scale = 1.0;

    if (!_ORIGINAL_VERSION) {
        // resize to fixed side length _TEMPLATE_SIZE to stabilize FPS
        if (_templateSz.height > _templateSz.width)
        { _scale = _templateSz.height / _TEMPLATE_SIZE; }
        else
        { _scale = _templateSz.width / _TEMPLATE_SIZE; }

        _templateSz = Size(floor(_templateSz.width / _scale), floor(_templateSz.height / _scale));
    }

    _baseTargetSz = Size(targetSize.width / _scale, targetSize.height / _scale);
    _templateScaleFactor = 1 / _scale;

    Size templateSzByCells = Size(floor((_templateSz.width) / _CELL_SIZE),
                                  floor((_templateSz.height) / _CELL_SIZE));

    // translation filter output target
    T outputSigma = sqrt(_templateSz.area() / ((1 + _PADDING) * (1 + _PADDING)))
                    * _OUTPUT_SIGMA_FACTOR / _CELL_SIZE;
    _y = gaussianShapedLabels2D<float>(outputSigma, templateSzByCells);
    calcDft(_y, _yf, 0);

    // translation filter hann window
    cv::Mat cosWindowX;
    cv::Mat cosWindowY;
    cosWindowY = hanningWindow<float>(_yf.rows);
    cosWindowX = hanningWindow<float>(_yf.cols);
    _cosWindow = cosWindowY * cosWindowX.t();

    std::shared_ptr<DFC> hfNum(new DFC);
    cv::Mat hfDen;

    if (getTranslationTrainingData(image, hfNum, hfDen, _pos) == false)
    { return false; }

    _hfNumerator = hfNum;
    _hfDenominator = hfDen;

    if (_scaleEstimator) {
        _scaleEstimator->reinit(image, _pos, targetSize,
                                _scale * _templateScaleFactor);
    }

    _lastBoundingBox = boundingBox;
    _isInitialized = true;
    return true;
}

bool DsstTracker::getTranslationTrainingData(const cv::Mat& image, std::shared_ptr<DFC>& hfNum,
        cv::Mat& hfDen, const Point& pos) const {
    std::shared_ptr<DFC> xt(new DFC);

    if (getTranslationFeatures(image, xt, pos, _scale) == false)
    { return false; }

    std::shared_ptr<DFC> xtf;

    if (_USE_CCS)
    { xtf = DFC::dftFeatures(xt); }
    else
    { xtf = DFC::dftFeatures(xt, cv::DFT_COMPLEX_OUTPUT); }

    hfNum = DFC::mulSpectrumsFeatures(_yf, xtf, true);
    hfDen = DFC::sumFeatures(DFC::mulSpectrumsFeatures(xtf, xtf, true));

    return true;
}

bool DsstTracker::getTranslationFeatures(const cv::Mat& image, std::shared_ptr<DFC>& features,
        const Point& pos, T scale) const {
    cv::Mat patch;
    Size patchSize = _templateSz * scale;

    if (getSubWindow(image, patch, patchSize, pos) == false)
    { return false; }

    if (_ORIGINAL_VERSION)
    { depResize(patch, patch, _templateSz); }
    else
    { resize(patch, patch, _templateSz, 0, 0, _RESIZE_TYPE); }

    cv::Mat floatPatch;
    patch.convertTo(floatPatch, CV_32FC(3));

    features.reset(new DFC());
    cvFhog(floatPatch, features, _CELL_SIZE, DFC::numberOfChannels() - 1);

    // append gray-scale image
    if (patch.channels() == 1) {
        if (_CELL_SIZE != 1)
        { resize(patch, patch, features->channels[0].size(), 0, 0, _RESIZE_TYPE); }

        features->channels[DFC::numberOfChannels() - 1] = patch / 255.0 - 0.5;
    } else {
        if (_CELL_SIZE != 1)
        { resize(patch, patch, features->channels[0].size(), 0, 0, _RESIZE_TYPE); }

        cv::Mat grayFrame;
        cvtColor(patch, grayFrame, cv::COLOR_BGR2GRAY);
        grayFrame.convertTo(grayFrame, CV_TYPE);
        grayFrame = grayFrame / 255.0 - 0.5;
        features->channels[DFC::numberOfChannels() - 1] = grayFrame;
    }

    DFC::mulFeatures(features, _cosWindow);
    return true;
}

bool DsstTracker::update_(const cv::Mat& image, Rect& boundingBox) {
    return updateAtScalePos(image, _pos, _scale, boundingBox);
}

bool DsstTracker::updateAt_(const cv::Mat& image, Rect& boundingBox) {
    bool isValid = false;
    T scale = 0;
    Point pos(boundingBox.x + boundingBox.width * consts::c0_5,
              boundingBox.y + boundingBox.height * consts::c0_5);

    // caller's box may have a different aspect ratio
    // compared to the _targetSize; use the larger side
    // to calculate scale
    if (boundingBox.width > boundingBox.height)
    { scale = boundingBox.width / _baseTargetSz.width; }
    else
    { scale = boundingBox.height / _baseTargetSz.height; }

    isValid = updateAtScalePos(image, pos, scale, boundingBox);
    return isValid;
}

 bool  DsstTracker::get_found(){
  return _FOUND;
}

bool DsstTracker::updateAtScalePos(const cv::Mat& image, const Point& oldPos, const T oldScale, Rect& boundingBox) {
    ++_frameIdx;

    if (!_isInitialized)
    { return false; }

    T newScale = oldScale;
    Point newPos = oldPos;
    cv::Point2i maxResponseIdx;
    cv::Mat response;

    // in case of error return the last box
    boundingBox = _lastBoundingBox;

    if (detectModel(image, response, maxResponseIdx, newPos, newScale) == false)
    { return false; }

    // return box
    Rect tempBoundingBox;
    tempBoundingBox.width = _baseTargetSz.width * newScale;
    tempBoundingBox.height = _baseTargetSz.height * newScale;
    tempBoundingBox.x = newPos.x - tempBoundingBox.width / 2;
    tempBoundingBox.y = newPos.y - tempBoundingBox.height / 2;

    _FOUND = evalReponse(image, response, maxResponseIdx, tempBoundingBox);


    if (updateModel(image, newPos, newScale) == false)
    { return false; }

    boundingBox &= Rect(0, 0, static_cast<int>(image.cols), static_cast<int>(image.rows));
    boundingBox = tempBoundingBox;
    _lastBoundingBox = tempBoundingBox;
    return true;
}

bool DsstTracker::evalReponse(const cv::Mat& image, const cv::Mat& response, const cv::Point2i& maxResponseIdx, const Rect& tempBoundingBox) const {
    T peakValue = 0;
    T psrClamped = calcPsr(response, maxResponseIdx, _PSR_PEAK_DEL, peakValue);

    if (psrClamped < _PSR_THRESHOLD)
    { return false; }

    // check if we are out of image, too small or too large
    Rect imageRect(Point(0, 0), image.size());
    Rect intersection = imageRect & tempBoundingBox;
    double  bbArea = tempBoundingBox.area();
    double areaThreshold = _MAX_AREA_FACTOR * imageRect.area();
    double intersectDiff = std::abs(bbArea - intersection.area());

    if (intersectDiff > 0.01 || bbArea < _MIN_AREA
            || bbArea > areaThreshold)
    { return false; }

    return true;
}

bool DsstTracker::detectModel(const cv::Mat& image, cv::Mat& response, cv::Point2i& maxResponseIdx, Point& newPos, T& newScale) const {
    // find translation
    std::shared_ptr<DFC> xt(new DFC);

    if (getTranslationFeatures(image, xt, newPos, newScale) == false)
    { return false; }

    std::shared_ptr<DFC> xtf;

    if (_USE_CCS)
    { xtf = DFC::dftFeatures(xt); }
    else
    { xtf = DFC::dftFeatures(xt, cv::DFT_COMPLEX_OUTPUT); }

    std::shared_ptr<DFC> sampleSpec = DFC::mulSpectrumsFeatures(_hfNumerator, xtf, false);
    cv::Mat sumXtf = DFC::sumFeatures(sampleSpec);
    cv::Mat hfDenLambda = addRealToSpectrum<float>(_LAMBDA, _hfDenominator);
    cv::Mat responseTf;

    if (_USE_CCS)
    { divSpectrums(sumXtf, hfDenLambda, responseTf, 0, false); }
    else
    { divideSpectrumsNoCcs<float>(sumXtf, hfDenLambda, responseTf); }

    cv::Mat translationResponse;
    idft(responseTf, translationResponse, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

    cv::Point delta;
    double maxResponse;
    cv::Point_<float> subDelta;
    minMaxLoc(translationResponse, 0, &maxResponse, 0, &delta);
    subDelta = delta;

    if (_CELL_SIZE != 1)
    { subDelta = subPixelDelta<float>(translationResponse, delta); }

    T posDeltaX = (subDelta.x + 1 - floor(translationResponse.cols / consts::c2_0)) * newScale;
    T posDeltaY = (subDelta.y + 1 - floor(translationResponse.rows / consts::c2_0)) * newScale;
    newPos.x += round(posDeltaX * _CELL_SIZE);
    newPos.y += round(posDeltaY * _CELL_SIZE);

    if (_scaleEstimator) {
        //find scale
        T tempScale = newScale * _templateScaleFactor;

        if (_scaleEstimator->detectScale(image, newPos,
                                         tempScale) == false)
        { return false; }

        newScale = tempScale / _templateScaleFactor;
    }

    response = translationResponse;
    maxResponseIdx = delta;
    return true;
}

bool DsstTracker::updateModel(const cv::Mat& image, const Point& newPos, T newScale) {
    _pos = newPos;
    _scale = newScale;
    std::shared_ptr<DFC> hfNum(new DFC);
    cv::Mat hfDen;

    if (getTranslationTrainingData(image, hfNum, hfDen, _pos) == false)
    { return false; }

    _hfDenominator = (1 - _LEARNING_RATE) * _hfDenominator + _LEARNING_RATE * hfDen;
    DFC::mulValueFeatures(_hfNumerator, (1 - _LEARNING_RATE));
    DFC::mulValueFeatures(hfNum, _LEARNING_RATE);
    DFC::addFeatures(_hfNumerator, hfNum);

    if (_scaleEstimator) {
        if (_scaleEstimator->updateScale(image, newPos, newScale * _templateScaleFactor) == false)
        { return false; }
    }

    return true;
}
