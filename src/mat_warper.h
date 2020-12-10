/**  
 *  Copyright (c) 2019 All Rights Reserved
 *  @author Zhao Chao
 *  @date 2019.06.06 15:01:21
 *  @brief decode to jpeg 
 */

#ifndef MAT_WARPER_H_
#define MAT_WARPER_H_

#include<opencv2/opencv.hpp>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py = pybind11;

cv::Mat numpy_uint8_1c_to_cv_mat(py::array_t<unsigned char>& input);

cv::Mat numpy_uint8_3c_to_cv_mat(py::array_t<unsigned char>& input);

py::array_t<unsigned char> cv_mat_uint8_1c_to_numpy(cv::Mat & input);

py::array_t<unsigned char> cv_mat_uint8_3c_to_numpy(cv::Mat  input);



#endif /* MAT_WARPER_H_ */
