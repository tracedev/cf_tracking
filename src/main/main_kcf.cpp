#include <tclap/CmdLine.h>
#include <iostream>
#include "kcf_tracker.hpp"
#include "tracker_run.hpp"

class KcfTrackerRun : public TrackerRun
{
public:
    KcfTrackerRun() : TrackerRun("KCFcpp")
    {}

    virtual ~KcfTrackerRun()
    {
    }

    virtual cf_tracking::CfTracker* parseTrackerParas(TCLAP::CmdLine& cmd, int argc, const char** argv)
    {
        cf_tracking::KcfParameters paras;
        TCLAP::SwitchArg originalVersion("", "original_version", "Parameters and performance as close to the KCF VOT version as possible.", cmd, false);
        TCLAP::SwitchArg originalParametersWithScaleFilter("", "original_parameters_scale_filter", "KCF VOT version parameters with DSST scale filter.", cmd, false);
        TCLAP::ValueArg<int> templateSize("", "para_template_size", "template size", false, paras.templateSize, "integer", cmd);
        TCLAP::ValueArg<int> cellSize("", "para_cell_size", "cell size of fhog", false, paras.cellSize, "integer", cmd);
        TCLAP::ValueArg<double> padding("", "para_padding", "padding around the target", false, paras.padding, "double", cmd);
        TCLAP::ValueArg<double> lambda("", "para_lambda", "regularization factor", false, paras.lambda, "double", cmd);
        TCLAP::ValueArg<double> outputSigmaFactor("", "para_output_sigma_factor", "spatial bandwidth of the target", false, paras.outputSigmaFactor, "double", cmd);
        TCLAP::ValueArg<double> scaleStep("", "para_vot_scale_step", "scale_step", false, paras.votScaleStep, "double", cmd);
        TCLAP::ValueArg<double> scaleWeight("", "para_vot_scale_weight", "scale_weight", false, paras.votScaleWeight, "double", cmd);
        TCLAP::ValueArg<double> interpFactor("", "para_interpFactor", "interpolation factor for learning", false, paras.interpFactor, "double", cmd);
        TCLAP::ValueArg<double> kernelSigma("", "para_kernel_sigma", "sigma for Gaussian kernel", false, paras.kernelSigma, "double", cmd);
        TCLAP::ValueArg<double> psrThreshold("", "para_psr_threshold", "if psr is lower than "
            "psr threshold, target is assumed to be lost", false, paras.psrThreshold, "double", cmd);
        TCLAP::ValueArg<int> psrPeakDel("", "para_psr_peak_del", "amount of pixels that are "
            "deleted for psr calculation around the peak (1 means that a window of 3 by 3 is "
            "deleted; 0 means that max response is deleted; 2 * peak_del + 1 pixels are deleted)", false, paras.psrPeakDel, "integer", cmd);
        TCLAP::SwitchArg useDsstScale("", "para_use_dsst_scale", "Uses the DSST scale filter for scale estimation. "
            "Disable for more speed!", cmd, paras.useDsstScaleEstimation);
        TCLAP::ValueArg<double> scaleSigmaFactor("", "para_dsst_sigma_factor", "DSST: spatial bandwidth of the target", false, paras.scaleSigmaFactor, "double", cmd);
        TCLAP::ValueArg<double> scaleEstimatorStep("", "para_dsst_scale_step", "DSST: scale step", false, paras.scaleEstimatorStep, "double", cmd);
        TCLAP::ValueArg<double> scaleLambda("", "para_dsst_lambda", "DSST: regularization for scale estimation", false, paras.scaleLambda, "double", cmd);
        TCLAP::ValueArg<int> scaleCellSize("", "para_dsst_cell_size", "DSST: hog cell size for scale estimation", false, paras.scaleCellSize, "integer", cmd);
        TCLAP::ValueArg<int> numberOfScales("", "para_dsst_scales", "DSST: number of scales", false, paras.numberOfScales, "integer", cmd);
        TCLAP::SwitchArg enableTrackingLossDetection("", "para_enable_tracking_loss", "Enables the tracking loss detection!", cmd, paras.enableTrackingLossDetection);

        cmd.parse(argc, argv);

        paras.padding = padding.getValue();
        paras.lambda = lambda.getValue();
        paras.outputSigmaFactor = outputSigmaFactor.getValue();
        paras.votScaleStep = scaleStep.getValue();
        paras.votScaleWeight = scaleWeight.getValue();
        paras.templateSize = templateSize.getValue();
        paras.interpFactor = interpFactor.getValue();
        paras.kernelSigma = kernelSigma.getValue();
        paras.cellSize = cellSize.getValue();
        paras.psrThreshold = psrThreshold.getValue();
        paras.psrPeakDel = psrPeakDel.getValue();
        paras.enableTrackingLossDetection = enableTrackingLossDetection.getValue();

        paras.useDsstScaleEstimation = useDsstScale.getValue();
        paras.scaleSigmaFactor = scaleSigmaFactor.getValue();
        paras.scaleEstimatorStep = scaleEstimatorStep.getValue();
        paras.scaleLambda = scaleLambda.getValue();
        paras.scaleCellSize = scaleCellSize.getValue();
        paras.numberOfScales = numberOfScales.getValue();

        if (originalVersion.getValue() || originalParametersWithScaleFilter.getValue())
        {
            paras.padding = 1.5;
            paras.lambda = 0.0001;
            paras.outputSigmaFactor = 0.1;
            paras.votScaleStep = 1.05;
            paras.votScaleWeight = 0.95;
            paras.templateSize = 100;
            paras.interpFactor = 0.012;
            paras.kernelSigma = 0.6;
            paras.cellSize = 4;
            paras.pixelPadding = 0;

            paras.enableTrackingLossDetection = false;

            if (originalParametersWithScaleFilter.getValue())
            {
                paras.useVotScaleEstimation = false;
                paras.useDsstScaleEstimation = true;
            }
            else
            {
                paras.useVotScaleEstimation = true;
                paras.useDsstScaleEstimation = false;
            }

            paras.useFhogTranspose = false;
        }

        return new cf_tracking::KcfTracker(paras);
    }
};

int main(int argc, const char** argv)
{
    KcfTrackerRun mainObj;

    if (!mainObj.start(argc, argv))
        return -1;

    return 0;
}
