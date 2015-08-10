# Introduction

This repository provides C++ implementations for two correlation filter-based trackers. The code implements modified versions of the
visual trackers proposed in [1] and [2]:
* KCFcpp: This tracker is a C++ port of the Matlab implementation of the kernelized correlation filter (KCF) tracker proposed in [1]. Project webpage:  http://home.isr.uc.pt/~henriques/circulant/ KCFcpp uses as default scale adaption the 1D scale filter proposed in [2]. In addition, a fixed template size,
the subpixel/subcell response peak estimation, and the model update from [3] is used as in the KCF version used by Henriques et al. in the VOT challenge 2014 (http://votchallenge.net/vot2014/). The scale adaption used by Henriques et al. in the VOT challenge 2014 is available as option.
* DSSTcpp: This tracker is a C++ port of the Matlab implementation of the discriminative scale space tracker (DSST) proposed in [2]. The default settings
use a fixed template size and the subpixel/cell response peak estimation as in
the KCF version. Project webpage: http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html

Both implementations use the FHOG features proposed in [4].

More specifically, the FHOG implementation  from [5] is used.

Both trackers offer the option to use the target loss detection proposed in [6].

# Commercial Use (US)
The code using linear correlation filters may be affected by a US patent. If you want to use this code commercially in the US please refer to http://www.cs.colostate.edu/~vision/ocof_toolset_2012/index.php for possible patent claims.

## 3rdparty libraries used:
* Piotr's Matlab Toolbox http://vision.ucsd.edu/~pdollar/toolbox/doc/
* OpenCV http://opencv.org/

## References
If you reuse this code for a scientific publication, please cite the related publications (dependent on what parts of the code you reuse):

[1]
```
@article{henriques2015tracking,
title = {High-Speed Tracking with Kernelized Correlation Filters},
author = {Henriques, J. F. and Caseiro, R. and Martins, P. and Batista, J.},
journal = {Pattern Analysis and Machine Intelligence, IEEE Transactions on},
year = {2015}
```


[2]
```
@inproceedings{danelljan2014dsst,
title={Accurate Scale Estimation for Robust Visual Tracking},
author={Danelljan, Martin and H{\"a}ger, Gustav and Khan, Fahad Shahbaz and Felsberg, Michael},
booktitle={Proceedings of the British Machine Vision Conference BMVC},
year={2014}}
```

[3]
```
@inproceedings{danelljan2014colorattributes,
title={Adaptive Color Attributes for Real-Time Visual Tracking},
author={Danelljan, Martin and Khan, Fahad Shahbaz and Felsberg, Michael and Weijer, Joost van de},
booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2014}}
```

[4]
```
@article{lsvm-pami,
title = "Object Detection with Discriminatively Trained Part Based Models",
author = "Felzenszwalb, P. F. and Girshick, R. B. and McAllester, D. and Ramanan, D.",
journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
year = "2010", volume = "32", number = "9", pages = "1627--1645"}
```

[5]
```
@misc{PMT,
author = {Piotr Doll\'ar},
title = {{P}iotr's {C}omputer {V}ision {M}atlab {T}oolbox ({PMT})},
howpublished = {\url{http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html}}}
```

[6]
```
@inproceedings{bolme2010mosse,
author={Bolme, David S. and Beveridge, J. Ross and Draper, Bruce A. and Yui Man Lui},
title={Visual Object Tracking using Adaptive Correlation Filters},
booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2010}}
```
