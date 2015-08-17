#include "math_helper.hpp"
#include <opencv2/imgproc/imgproc.hpp>

namespace cf_tracking {
    int mod(int dividend, int divisor) {
        // http://stackoverflow.com/questions/12276675/modulus-with-negative-numbers-in-c
        return ((dividend % divisor) + divisor) % divisor;
    }

    void dftCcs(const cv::Mat& input, cv::Mat& out, int flags) {
        cv::dft(input, out, flags);
    }

    void dftNoCcs(const cv::Mat& input, cv::Mat& out, int flags) {
        flags = flags | cv::DFT_COMPLEX_OUTPUT;
        cv::dft(input, out, flags);
    }

    // use bi-linear interpolation on zoom, area otherwise
    // similar to mexResize.cpp of DSST
    // http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html
    void depResize(const cv::Mat& source, cv::Mat& dst, const cv::Size& dsize) {
        int interpolationType = cv::INTER_AREA;

        if (dsize.width > source.cols
                || dsize.height > source.rows)
        { interpolationType = cv::INTER_LINEAR; }

        cv::resize(source, dst, dsize, 0, 0, interpolationType);
    }
}
