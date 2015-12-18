#include <algorithm>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <string>

#include "caffe/neuron_layers.hpp"
#include <google/protobuf/repeated_field.h>


int batchSize = 100;

namespace caffe {

template <typename Dtype>
void loadParamArray(std::vector<Dtype> &data, google::protobuf::RepeatedField<float> &param, std::string nameOfParam, const int count){
  google::protobuf::RepeatedField<float> data_proto = param;
  // std::vector<Dtype> data(data_proto.size());
  int counter = 0;
  // std::cout << "data_proto.size():" << data_proto.size() << std::endl;
  // std::cout << "data.size():" << data.size() << std::endl;
  for (google::protobuf::RepeatedField<float>::iterator it = data_proto.begin();
    it != data_proto.end(); ++it) {
    data[counter] = *it;
    counter++;
  }
  if (count != data.size()) {
    std::cout << "Size mismatch : there are " << count << " inputs, but " << nameOfParam << " provided containes " << data.size() << " values. Thus only " << count << " first values will be used." << std::endl;
  }
}

/**
 * @brief Modifies values with mean values
 */
template <typename Dtype>
void TweakFeaturesLayer<Dtype>::alterValues(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  std::cout << "alter mode!!" << std::endl;
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = bottom[0]->count();

  int mode = this->layer_param_.tweak_features_param().mode();
  double factor = this->layer_param_.tweak_features_param().factor();
  std::cout << "alter mode 2" << std::endl;
  int add_value = this->layer_param_.tweak_features_param().add_value();

  // get the features_mean values from prototxt parameter
  google::protobuf::RepeatedField<double> features_mean_proto = this->layer_param_.tweak_features_param().features_mean();
  std::vector<Dtype> mean(features_mean_proto.size());
  int counter = 0;
  // std::cout << "features_mean_proto.size():" << features_mean_proto.size() << std::endl;
  // std::cout << "mean.size():" << mean.size() << std::endl;
  for (google::protobuf::RepeatedField<double>::iterator it = features_mean_proto.begin();
    it != features_mean_proto.end(); ++it) {
    mean[counter] = *it;
    counter++;
  }
  if (count != mean.size() * batchSize) {
    std::cout << "Size mismatch : there are " << count << " inputs, but features_mean provided containes " << mean.size() << " values. Thus only " << count << " first values will be exagerated." << std::endl;
  }

  // std::vector<Dtype> mean = loadParamArray<Dtype>(this->layer_param_.tweak_features_param().features_mean(), "feature_mean", count);

  // modify input data depending on mean
  std::cout << "exageration factor:" << factor << std::endl;
  for (int i = 0; i < count; ++i) {
    // top_data[i] = bottom_data[i] + // original value
                  // add_value; // constant
    // if ((count % batchSize) < mean.size()) {
      // top_data[i] -= (mean[i % batchSize] - bottom_data[i]) * factor; // exagerated difference
      std::cout << "D" << i;
      top[0]->mutable_cpu_data()[i] = bottom[0]->cpu_data()[i] + (mean[i % batchSize] - bottom[0]->cpu_data()[i]) * factor;
    // }
    // top[0]->mutable_cpu_data()[i] = bottom[0]->cpu_data()[i];
  }
}


// /**
//  * @brief saves the values in order to compute the mean required by alterValues.
//  * Writes the sums so far.
//  */
// template <typename Dtype>
// void TweakFeaturesLayer<Dtype>::saveAverageValues(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {
//   std::cout << "save mode" << std::endl;
//   static int sampleCounter = 0;
//   static std::ofstream myfile;
//   static std::vector<Dtype> sums; // each element [i] stores the sum of bottom_data[i]

//   const Dtype* bottom_data = bottom[0]->cpu_data();
//   const int count = bottom[0]->count();

//   // if this is the first sample
//   if (sampleCounter == 0) {
//     sums.resize(count); // resize the sums vector properly
//   }
//   myfile.open(this->layer_param_.tweak_features_param().output_file_name().c_str(),  std::ofstream::out | std::ofstream::trunc); // open output file
//   sampleCounter++;


//   // create a converter from float to string
//   // std::ostringstream strs;

//   // write average of previous and current bottom_data to file
//   for (int i = 0; i < count; ++i) {
//     sums[i] += bottom_data[i];
//     myfile << sums[i] / sampleCounter << (i == count-1 ? "\n" : ", ");
//   }
//   myfile.close();
// }

/**
 * @brief saves the values in order to compute the mean required by alterValues.
 * Writes the sums so far.
 */
template <typename Dtype>
void TweakFeaturesLayer<Dtype>::saveValues(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  std::cout << "save mode for batch size of " << batchSize << std::endl;
  static int sampleCounter = 0;
  static std::ofstream myfile;
  static std::vector<Dtype> sums; // each element [i] stores the sum of bottom_data[i]

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const int count = bottom[0]->count();

  // if this is the first sample
  if (sampleCounter == 0) {
    myfile.open(this->layer_param_.tweak_features_param().output_file_name().c_str()); // open output file
    sums.resize(count); // resize the sums vector properly
  }
  sampleCounter++;


  // create a converter from float to string
  // std::ostringstream strs;

  // write bottom_data to file
  for (int i = 0; i < count; ++i) {
    sums[i] += bottom_data[i];
    if (i % batchSize == 0) {
      myfile << std::endl;
    }
    myfile << sums[i] / sampleCounter << (i == count-1 ? "\n" : ", ");
  }
}

/**
 * @brief called by caffe. Depending on parameter, saves or alters
 */
template <typename Dtype>
void TweakFeaturesLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Modes : 0: fast forward (does not alter the data). 1: save. 2: modify
  int mode = this->layer_param_.tweak_features_param().mode();
  if (mode == 0) { // fast forward mode
  } else if (mode == 1) { // save mode
    this->saveValues(bottom, top);
  } else if (mode >= 2) { // alter mode
    std::cout << "Entering alterValues mode" << std::endl;
    this->alterValues(bottom, top);
  }
}

template <typename Dtype>
void TweakFeaturesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Not implemented because useless for the project.
}


#ifdef CPU_ONLY
STUB_GPU(TweakFeaturesLayer);
#endif

INSTANTIATE_CLASS(TweakFeaturesLayer);
REGISTER_LAYER_CLASS(TWEAK_FEATURES, TweakFeaturesLayer);

}  // namespace caffe

