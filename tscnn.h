#pragma once

#include "strf.h"

#ifdef TSCNN_CUDA
#include <nvrtc.h>
#include <cuda.h>
#endif

#include <memory>
#include <functional>
#include <vector>
#include <stdexcept>
#include <random>
#include <unordered_map>

namespace tscnn {

#ifdef TSCNN_CUDA
static void check_nvrtc_debug(nvrtcResult err, const char* file, int line) {
	if (err == NVRTC_SUCCESS) return;
	throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + nvrtcGetErrorString(err));
}

static void check_cu_debug(CUresult err, const char* file, int line) {
	if (err == CUDA_SUCCESS) return;
	const char* str = "unknown error";
	cuGetErrorString(err, &str);
	throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + std::string("cuda error ") + str);
}

#define check_nvrtc(x) check_nvrtc_debug(x, __FILE__, __LINE__);
#define check_cu(x) check_cu_debug(x, __FILE__, __LINE__);

#endif

	template<typename...T>
	std::string format(const char*fmt, T&&... args) {
		std::string str;
		strf::format(str, fmt, std::forward<T>(args)...);
		return str;
	}

	static std::string replace(std::string str, std::vector<std::pair<std::string, std::string>> vars) {
		std::sort(vars.begin(), vars.end(), [](auto& a, auto& b) {
			return a.first.size() > b.first.size();
		});
		for (auto& v : vars) {
			while (true) {
				auto i = str.find(v.first);
				if (i == std::string::npos) break;
				str.replace(i, v.first.size(), v.second);
			}
		}
		return str;
	}

	struct vector_ref {
		size_t offset;
		size_t size;
		vector_ref select(size_t select_offset, size_t select_size) {
			return { offset + select_offset, select_size };
		}
	};

	struct gradients_index_ref {
		size_t index;
		size_t offset;
		size_t size;
		size_t allocation_size;
	};

	struct unit_ref {
		vector_ref output;
		gradients_index_ref gradients_index;
	};

	template<typename value_t = float, typename allocator_T = std::allocator<void>>
	struct nn {

		template<typename T>
		using a_vector = std::vector<T, typename std::allocator_traits<allocator_T>::template rebind_alloc<T>>;

		template<typename T>
		using a_function = std::function<T>;

		a_function<void(nn&, value_t*)> forward = [](nn&, value_t*) {};
		a_function<void(nn&, value_t*, value_t*)> backward = [](nn&, value_t*, value_t*) {};

		a_function<std::string(nn&, int)> gen_cuda_forward = [](nn&, int) {return "";};
		a_function<std::string(nn&, int)> gen_cuda_backward = [](nn&, int) {return "";};

		a_vector<a_function<void(nn&)>> construct_funcs;
		bool constructed = false;

		a_vector<a_function<void(nn&, const a_function<void(size_t weight_offset, size_t weight_n, size_t bias_offset, size_t bias_n, size_t inputs, size_t outputs)>& func)>> init_weights_funcs;

		a_vector<value_t> values;
		size_t total_weights = 0;

		a_vector<std::pair<gradients_index_ref, a_vector<std::pair<vector_ref, gradients_index_ref>>>> gradients;
		a_vector<bool> gradient_increment_first;

		std::minstd_rand rng_e;

		bool is_training = false;
		bool is_evaluating = false;

		void set_training() {
			is_training = true;
			is_evaluating = false;
		}
		void set_evaluating() {
			is_training = false;
			is_evaluating = true;
		}

		size_t values_batch_size = 0;
		size_t batch_size = 0;

		void set_batch_size(size_t n) {
			batch_size = n;
			values_batch_size = values.size();
			values.resize(values.size() * n);
		}

		vector_ref new_vector_ref(size_t size) {
			if ((uintptr_t)(values.data() + values.size()) % 4 != 0) values.resize(((uintptr_t)(values.data() + values.size()) & -4) + 4 - (uintptr_t)values.data());
			size_t offset = values.size();
			values.resize(offset + size);
			return { offset, size };
		}

		value_t* get_values(vector_ref ref) {
			return values.data() + ref.offset;
		}

		vector_ref get_input_gradient(unit_ref u) {
			return gradients[u.gradients_index.index].second[0].first;
		}

		std::vector<vector_ref> cuda_vars;
		std::unordered_map<size_t, size_t> cuda_var_last_use;

		std::unordered_map<size_t, std::string> cuda_value_var;
		std::string set_cuda_value_var(vector_ref ref, std::string name) {
			cuda_value_var[ref.offset] = name;
			cuda_vars.push_back(ref);
			cuda_var_last_use[ref.offset] = cuda_vars.size() - 1;
			//printf("set cuda value %d %d  %s  last use %d\n", ref.offset, ref.size, name.c_str(), cuda_var_last_use[ref.offset]);
			return format("value_t* %s = ${VAR_%d};${SYNC_VAR_%d}\n", name, ref.offset, ref.offset);
		}

		std::string gen_cuda_get_values(vector_ref ref, bool allow_var = true) {
			if (allow_var) {
				auto i = cuda_value_var.find(ref.offset);
				if (i != cuda_value_var.end()) {
					//printf("get cuda value %d %d  %s  last use %d\n", ref.offset, ref.size, i->second.c_str(), cuda_var_last_use[ref.offset]);
					cuda_var_last_use[ref.offset] = cuda_vars.size() - 1;
					return i->second;
				}
			}
			return format("(values + %d)", ref.offset);
		}

		bool need_cuda_sync = false;

		void reset_cuda_gen() {
			need_cuda_sync = false;
			cuda_vars.clear();
			cuda_var_last_use.clear();
		}

		std::string allocate_cuda_vars(std::string str) {
			std::vector<std::pair<std::string, std::string>> replacements;
			std::vector<std::tuple<size_t, size_t, size_t>> allocated;
			size_t size = 0;
			for (size_t ci = 0; ci != cuda_vars.size(); ++ci) {
				auto& v = cuda_vars[ci];
				size_t i = 0;
				bool removed_something = false;
				for (auto it = allocated.begin(); it != allocated.end();) {
					size_t from = std::get<0>(*it);
					size_t to= std::get<1>(*it);
					size_t offset = std::get<2>(*it);
					if (cuda_var_last_use[offset] < ci) {
						//printf("free [%d,%d) because %d <= %d\n", from, to, cuda_var_last_use[offset], ci);
						it = allocated.erase(it);
						removed_something = true;
						continue;
					}
					if (from - i >= v.size) {
						//printf("allocate %d to [%d,%d)\n", offset, i, i + v.size);
						allocated.emplace(it, i, i + v.size, v.offset);
						replacements.emplace_back(format("${VAR_%d}", v.offset), format("&vars[%d]", i));
						replacements.emplace_back(format("${SYNC_VAR_%d}", v.offset), removed_something ? "__syncthreads();" : "");
						i = -1;
						break;
					}
					i = to;
					++it;
				}
				if (i != -1) {
					if (i + v.size > 1024 * 48 / sizeof(value_t)) {
						replacements.emplace_back(format("${VAR_%d}", v.offset), gen_cuda_get_values(v, false));
						replacements.emplace_back(format("${SYNC_VAR_%d}", v.offset), removed_something ? "__syncthreads();" : "");
					} else {
						size = std::max(size, i + v.size);
						//printf("allocate %d to [%d,%d)\n", v.offset, i, i + v.size);
						allocated.emplace_back(i, i + v.size, v.offset);
						replacements.emplace_back(format("${VAR_%d}", v.offset), format("&vars[%d]", i));
						replacements.emplace_back(format("${SYNC_VAR_%d}", v.offset), removed_something ? "__syncthreads();" : "");
					}
				}
			}
			replacements.emplace_back("${SHARED_MEM}", size == 0 ? "" : format("__shared__ value_t vars[%d];\n", size));
			//std::quick_exit(0);
			return replace(str, replacements);
		}

		std::string cuda_check_sync() {
			if (need_cuda_sync) {
				need_cuda_sync = false;
				return "__syncthreads();\n";
			} else return {};
		}

		void cuda_set_sync() {
			need_cuda_sync = true;
		}

		void construct() {
			if (constructed) throw std::runtime_error("already constructed");
			constructed = true;
			for (auto& v : construct_funcs) {
				v(*this);
			}

			for (auto& v : gradients) {
				auto& vec = v.second;
				if (vec.empty()) continue;

				if (vec[0].first.size == v.first.size) {
					gradient_increment_first.push_back(true);
				} else {
					gradient_increment_first.push_back(false);
					new_gradient(v.first);
					std::swap(vec.front(), vec.back());
				}

			}

			for (auto& v : values) v = std::numeric_limits<value_t>::signaling_NaN();
		}

		int cuda_threads = 128;

		std::string gen_cuda_forward_kernel() {
			std::string type = "unknown-type";
			if (std::is_same<value_t, float>::value) type = "float";
			if (std::is_same<value_t, double>::value) type = "double";
			std::string str;
			str += "typedef " + type + " value_t;\n";
			str += "extern \"C\" __global__ void forward(value_t* values, value_t* in_weights) {\n";
			str += "int thread_index = threadIdx.x;\n";
			str += "int batch_index = blockIdx.x;\n";
			str += "if (batch_index >= " + std::to_string(batch_size) + ") return;\n";
			str += "values += " + std::to_string(values_batch_size) + " * blockIdx.x;\n";
			reset_cuda_gen();
			str += "${SHARED_MEM}";
			str += gen_cuda_forward(*this, cuda_threads);

			str += "\n}";
			str = allocate_cuda_vars(str);
			return str;
		}

		template<typename criterion_T>
		std::string gen_cuda_forward_backward_kernel(criterion_T&& criterion) {
			std::string type = "unknown-type";
			if (std::is_same<value_t, float>::value) type = "float";
			if (std::is_same<value_t, double>::value) type = "double";
			std::string str;
			str += "typedef " + type + " value_t;\n";
			str += "extern \"C\" __global__ void forward_backward(value_t* values, value_t* in_weights, value_t* grad_in, value_t* criterion_input, int criterion_input_size, value_t* criterion_target, value_t* criterion_loss, value_t* criterion_gradient) {\n";
			str += "int thread_index = threadIdx.x;\n";
			str += "int batch_index = blockIdx.x;\n";
			str += "if (batch_index >= " + std::to_string(batch_size) + ") return;\n";
			str += "values += " + std::to_string(values_batch_size) + " * batch_index;\n";
			str += "grad_in += " + std::to_string(values_batch_size) + " * batch_index;\n";
			str += "criterion_input += " + std::to_string(values_batch_size) + " * batch_index;\n";
			str += "criterion_target += " + std::to_string(values_batch_size) + " * batch_index;\n";
			str += "criterion_loss += " + std::to_string(values_batch_size) + " * batch_index;\n";
			str += "criterion_gradient += " + std::to_string(values_batch_size) + " * batch_index;\n";
			reset_cuda_gen();
			str += "${SHARED_MEM}";
			str += gen_cuda_forward(*this, cuda_threads);
			str += criterion.gen_cuda_forward(*this, cuda_threads);
			str += criterion.gen_cuda_backward(*this, cuda_threads);
			str += gen_cuda_backward(*this, cuda_threads);

			str += "\n}";
			str = allocate_cuda_vars(str);
			return str;
		}

		void init_weights(const a_function<void(size_t weight_offset, size_t weight_n, size_t bias_offset, size_t bias_n, size_t inputs, size_t outputs)>& func) {
			for (auto& v : init_weights_funcs) {
				v(*this, func);
			}
		}

		gradients_index_ref new_gradients_index(size_t size) {
			gradients.emplace_back();
			return gradients.back().first = { gradients.size() - 1, 0, size, size };
		}

		gradients_index_ref select_gradients_index(gradients_index_ref index, size_t offset, size_t size) {
			return { index.index, offset, size, index.size };
		}

		value_t* combine_gradients(gradients_index_ref index) {
			auto& grads = gradients[index.index].second;
			if (grads.empty()) throw std::runtime_error("missing gradients (some output is left unconnected ?)");
			size_t size = index.size;
			if (grads[0].first.size != size) throw std::runtime_error("base gradients size mismatch");
			value_t* dst = get_values(grads[0].first);
			if (gradient_increment_first[index.index]) {
				for (size_t i = 1; i < grads.size(); ++i) {
					size_t offset = grads[i].second.offset;
					size_t size2 = grads[i].second.size;
					value_t* src = get_values(grads[i].first);
					for (size_t i2 = 0; i2 < size2; ++i2) {
						dst[offset + i2] += src[i2];
					}
				}
			} else {
				for (size_t i2 = 0; i2 < size; ++i2) {
					dst[i2] = 0.0;
				}
				for (size_t i = 1; i < grads.size(); ++i) {
					size_t offset = grads[i].second.offset;
					size_t size2 = grads[i].second.size;
					value_t* src = get_values(grads[i].first);
					for (size_t i2 = 0; i2 < size2; ++i2) {
						dst[offset + i2] += src[i2];
					}
				}
			}
			return dst;
		}

		std::string gen_cuda_combine_gradients(gradients_index_ref index) {
			auto& grads = gradients[index.index].second;
			if (grads.empty()) throw std::runtime_error("missing gradients (some output is left unconnected ?)");
			size_t size = index.size;
			if (grads[0].first.size != size) throw std::runtime_error("base gradients size mismatch");
			std::string str = "value_t* gradients = " + gen_cuda_get_values(grads[0].first) + ";\n";
			std::string op = gradient_increment_first[index.index] ? "+=" : "=";
			if (!gradient_increment_first[index.index] || grads.size() > 1) {
				str += "{\n";
				str += "value_t* dst = gradients;\n";
				for (size_t i = 1; i < grads.size(); ++i) {
					str += "value_t* src" + std::to_string(i) + " = " + gen_cuda_get_values(grads[i].first) + ";\n";
				}
				std::vector<size_t> adds;
				std::vector<size_t> prev_adds;
				size_t i2 = 0;
				size_t prev_i2 = 0;
				auto flush = [&]() {
					if (!adds.empty() || !gradient_increment_first[index.index]) {
						str += format("for (size_t i = 0; i != %d; ++i) *dst++ %s ", i2 - prev_i2, op);
						if (adds.empty()) str += "0";
						else {
							bool first = true;
							for (size_t i : adds) {
								if (!first) str += " + ";
								first = false;
								str += format("*src%d++", i);
							}
						}
						str += ";\n";
					}
					prev_i2 = i2;
				};
				for (; i2 < size; ++i2) {
					adds.clear();
					for (size_t i = 1; i < grads.size(); ++i) {
						size_t offset = grads[i].second.offset;
						size_t size2 = grads[i].second.size;
						if (i2 >= offset && i2 < offset + size2) adds.push_back(i);
					}
					if (!prev_adds.empty() && adds != prev_adds) {
						flush();
						prev_adds = adds;
					}
				}
				flush();
				str += "}\n";
			}
			return str;
		}

		vector_ref new_gradient(gradients_index_ref index) {
			vector_ref r = new_vector_ref(index.size);
			gradients[index.index].second.emplace_back(r, index);
			//r.offset += index.offset;
			//r.size = index.size;
			return r;
		}

		a_vector<unit_ref> inputs;
		a_vector<unit_ref> outputs;

		unit_ref make_input(size_t out) {
			gradients_index_ref gradients_index = new_gradients_index(out);
			construct_funcs.push_back([gradients_index](nn& n) {
				auto parent_backward = n.backward;
				n.backward = [gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					n.combine_gradients(gradients_index);
					parent_backward(n, in_weights, grad_in);
				};
				auto parent_gen_cuda_backward = n.gen_cuda_backward;
				n.gen_cuda_backward = [gradients_index, parent_gen_cuda_backward](nn& n, int cuda_threads) {
					std::string str = replace(R"(
					{
						$combine_gradients
					}
					)",
					{
						{"$combine_gradients", n.gen_cuda_combine_gradients(gradients_index)}
					});

					return str + parent_gen_cuda_backward(n, cuda_threads);
				};
			});
			unit_ref r = { new_vector_ref(out), gradients_index };
			inputs.push_back(r);
			return r;
		}

		unit_ref make_output(unit_ref input_unit_ref) {
			vector_ref input_ref = input_unit_ref.output;
			vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
			vector_ref output_ref = input_ref;
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			construct_funcs.push_back([input_ref, input_gradients_ref, output_ref, gradients_index](nn& n) {
				auto parent_backward = n.backward;
				n.backward = [input_ref, input_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input = n.get_values(input_ref);
					value_t* input_gradients = n.get_values(input_gradients_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					for (size_t i = 0; i < input_gradients_ref.size; ++i) {
						if (std::isnan(gradients[i])) throw std::runtime_error("output gradient is nan");
						input_gradients[i] = gradients[i];
					}
					parent_backward(n, in_weights, grad_in);
				};
				auto parent_gen_cuda_forward = n.gen_cuda_forward;
				n.gen_cuda_forward = [input_ref, output_ref, parent_gen_cuda_forward](nn& n, int cuda_threads) {
					std::string str = parent_gen_cuda_forward(n, cuda_threads);

					str += n.cuda_check_sync();

					str += replace(R"(
					{
						for (size_t i = 0; i < $input_size; i += $threads) {
							$output[i] = $input[i];
						}
					}
					)",
					{
						{"$threads", std::to_string(cuda_threads)},
						{"$input", n.gen_cuda_get_values(input_ref)},
						{"$output", n.gen_cuda_get_values(output_ref, false)},
						{"$input_size", std::to_string(input_ref.size)},
					});

					n.cuda_set_sync();

					return str;
				};
				auto parent_gen_cuda_backward = n.gen_cuda_backward;
				n.gen_cuda_backward = [input_ref, input_gradients_ref, gradients_index, parent_gen_cuda_backward](nn& n, int cuda_threads) {
					std::string str;

					std::string gradval = "grad" + std::to_string(input_gradients_ref.offset);
					str += n.set_cuda_value_var(input_gradients_ref, gradval);

					str += replace(R"(
					{
						value_t* input_gradients = $input_gradients;
						$combine_gradients
						for (size_t i = 0; i < $input_size; i += $threads) {
							input_gradients[i] = gradients[i];
						}
					}
					)",
					{
						{"$threads", std::to_string(cuda_threads)},
						{"$input_gradients", n.gen_cuda_get_values(input_gradients_ref)},
						{"$input_size", std::to_string(input_ref.size)},
						{"$combine_gradients", n.gen_cuda_combine_gradients(gradients_index)}
					});

					n.cuda_set_sync();

					return str + parent_gen_cuda_backward(n, cuda_threads);
				};
			});
			unit_ref r = { output_ref, gradients_index };
			outputs.push_back(r);
			return r;
		}

		unit_ref make_linear(size_t out, unit_ref input_unit_ref) {
			vector_ref input_ref = input_unit_ref.output;
			vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
			vector_ref output_ref = new_vector_ref(out);
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			size_t weights = output_ref.size + input_ref.size * output_ref.size;
			size_t weights_offset = total_weights;
			//log("linear has %d weights at offset %d\n", weights, weights_offset);
			total_weights += weights;
			init_weights_funcs.push_back([input_ref, output_ref, weights_offset](nn&, const a_function<void(size_t weight_offset, size_t weight_n, size_t bias_offset, size_t bias_n, size_t inputs, size_t outputs)>& func) {
				size_t input_size = input_ref.size;
				size_t output_size = output_ref.size;
				func(weights_offset + output_size, output_size * input_size, weights_offset, output_size, input_size, output_size);
			});
			construct_funcs.push_back([input_ref, input_gradients_ref, output_ref, gradients_index, weights_offset](nn& n) {
				auto parent_forward = n.forward;
				n.forward = [input_ref, output_ref, weights_offset, parent_forward](nn& n, value_t* in_weights) {
					parent_forward(n, in_weights);
					value_t* input = n.get_values(input_ref);
					value_t* output = n.get_values(output_ref);
					size_t input_size = input_ref.size;
					size_t output_size = output_ref.size;
					value_t* bias_w = in_weights + weights_offset;
					value_t* w = bias_w + output_size;
					for (size_t oi = 0; oi < output_size; ++oi) {
						output[oi] = *bias_w++;
						for (size_t ii = 0; ii < input_size; ++ii) {
							output[oi] += input[ii] * *w++;
						}
					}
				};
				auto parent_backward = n.backward;
				n.backward = [input_ref, input_gradients_ref, output_ref, gradients_index, weights_offset, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input = n.get_values(input_ref);
					value_t* input_gradients = n.get_values(input_gradients_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					size_t input_size = input_ref.size;
					size_t output_size = output_ref.size;
					value_t* bias_w = in_weights + weights_offset;
					value_t* w = bias_w + output_size;
					for (size_t ii = 0; ii < input_size; ++ii) {
						input_gradients[ii] = 0;
						value_t* column = w;
						for (size_t oi = 0; oi < output_size; ++oi) {
							input_gradients[ii] += gradients[oi] * *w;
							w += input_size;
						}
						w = column + 1;
					}
					value_t* bias_g = grad_in + weights_offset;
					value_t* g = bias_g + output_size;
					for (size_t oi = 0; oi < output_size; ++oi) {
						*bias_g++ += gradients[oi];
						for (size_t ii = 0; ii < input_size; ++ii) {
							*g++ += gradients[oi] * input[ii];
						}
					}
					parent_backward(n, in_weights, grad_in);
				};
				auto parent_gen_cuda_forward = n.gen_cuda_forward;
				n.gen_cuda_forward = [input_ref, output_ref, weights_offset, parent_gen_cuda_forward](nn& n, int cuda_threads) {
					std::string str = parent_gen_cuda_forward(n, cuda_threads);

					std::string outval = "out" + std::to_string(output_ref.offset);
					str += n.set_cuda_value_var(output_ref, outval);

					str += n.cuda_check_sync();

					str += replace(R"(
					{
						value_t* input = $input;
						value_t* output = $output;
						const size_t input_size = $input_size;
						const size_t output_size = $output_size;
						value_t* bias_w = in_weights + $weights_offset;
						value_t* w = bias_w + output_size;
						bias_w += thread_index;
						w += $input_size * thread_index;

						for (size_t oi = thread_index; oi < output_size; oi += $threads, bias_w += $threads - 1, w += ($threads - 1) * input_size) {
							value_t sum = *bias_w++;
							for (size_t ii = 0; ii < input_size; ++ii) {
								sum += input[ii] * *w++;
							}
							output[oi] = sum;
						}

//						int len = (output_size + $threads - 1) / $threads;
//						value_t* output_end = output + output_size;
//						output += thread_index;
//						int spin = min(max(int(output + len - output_end), 0), len);
//						len -= spin;
//						//printf("%d: len %d  spin %d  output_size %d\n", thread_index, len, spin, (int)output_size);
//						while (len--) {
//							__syncthreads();
//							value_t tmp = *bias_w++;
//							for (size_t ii = 0; ii < input_size; ++ii) {
//								tmp += input[ii] * *w++;
//							}
//							*output = tmp;
//							output+= $threads;
//							bias_w += $threads - 1;
//							w += ($threads - 1) * input_size;
//						}
//						while (spin--) {
//							__syncthreads();
//						}

//						int len = (output_size + $threads - 1) / $threads;
//						int offset = len * thread_index;
//						if (offset + len > output_size) len = output_size - offset;
//						//printf("%d: at $output offset %d len %d  for output_size %d\n", thread_index, offset, len, (int)output_size);
//						bias_w += offset;
//						w += offset * input_size;
//						output += offset;
//						for (;len > 0;--len) {
//							value_t tmp = *bias_w++;
//							for (size_t ii = 0; ii < input_size; ++ii) {
//								tmp += input[ii] * *w++;
//							}
//							*output++ = tmp;
//						}
//						__syncthreads();
					}
					)",
					{
						{"$threads", std::to_string(cuda_threads)},
						{"$input", n.gen_cuda_get_values(input_ref)},
						{"$output", n.gen_cuda_get_values(output_ref)},
						{"$input_size", std::to_string(input_ref.size)},
						{"$output_size", std::to_string(output_ref.size)},
						{"$weights_offset", std::to_string(weights_offset)},
					});

					n.cuda_set_sync();

					return str;
				};
				auto parent_gen_cuda_backward = n.gen_cuda_backward;
				n.gen_cuda_backward = [input_ref, input_gradients_ref, output_ref, gradients_index, weights_offset, parent_gen_cuda_backward](nn& n, int cuda_threads) {

					std::string str;

					std::string gradval = "grad" + std::to_string(input_gradients_ref.offset);
					str += n.set_cuda_value_var(input_gradients_ref, gradval);

					str += n.cuda_check_sync();

					str += replace(R"(
					{
						value_t* input = $input;
						value_t* input_gradients = $input_gradients;
						$combine_gradients
						size_t input_size = $input_size;
						size_t output_size = $output_size;
						value_t* w = in_weights + $weights_offset + output_size;
						w += thread_index;
						for (size_t ii = thread_index; ii < input_size; ii += $threads) {
							input_gradients[ii] = 0;
							value_t* column = w;
							for (size_t oi = 0; oi < output_size; ++oi) {
								input_gradients[ii] += gradients[oi] * *w;
								w += input_size;
							}
							w = column + $threads;
						}
						value_t* bias_g = grad_in + $weights_offset;
						value_t* g = bias_g + output_size;
						bias_g += thread_index;
						g += $input_size * thread_index;
						for (size_t oi = thread_index; oi < output_size; oi += $threads, bias_g += $threads - 1, g += ($threads - 1) * input_size) {
							*bias_g++ += gradients[oi];
							for (size_t ii = 0; ii < input_size; ++ii) {
								*g++ += gradients[oi] * input[ii];
							}
						}
					}
					)",
					{
						{"$threads", std::to_string(cuda_threads)},
						{"$input", n.gen_cuda_get_values(input_ref)},
						{"$input_gradients", n.gen_cuda_get_values(input_gradients_ref)},
						{"$input_size", std::to_string(input_ref.size)},
						{"$output_size", std::to_string(output_ref.size)},
						{"$weights_offset", std::to_string(weights_offset)},
						{"$combine_gradients", n.gen_cuda_combine_gradients(gradients_index)}
					});

					n.cuda_set_sync();

					return str + parent_gen_cuda_backward(n, cuda_threads);
				};
			});
			return { output_ref, gradients_index };
		}

		unit_ref make_sigmoid(unit_ref input_unit_ref) {
			vector_ref input_ref = input_unit_ref.output;
			vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
			vector_ref output_ref = new_vector_ref(input_ref.size);
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			construct_funcs.push_back([input_ref, input_gradients_ref, output_ref, gradients_index](nn& n) {
				auto parent_forward = n.forward;
				n.forward = [input_ref, output_ref, parent_forward](nn& n, value_t* in_weights) {
					parent_forward(n, in_weights);
					value_t* input = n.get_values(input_ref);
					value_t* output = n.get_values(output_ref);
					for (size_t i = 0; i < output_ref.size; ++i) {
						output[i] = (value_t)1.0 / ((value_t)1.0 + std::exp(-input[i]));
					}
				};
				auto parent_backward = n.backward;
				n.backward = [input_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input_gradients = n.get_values(input_gradients_ref);
					value_t* output = n.get_values(output_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					for (size_t i = 0; i < input_gradients_ref.size; ++i) {
						input_gradients[i] = gradients[i] * ((value_t)1.0 - output[i]) * output[i];
					}
					parent_backward(n, in_weights, grad_in);
				};
				auto parent_gen_cuda_forward = n.gen_cuda_forward;
				n.gen_cuda_forward = [input_ref, output_ref, parent_gen_cuda_forward](nn& n, int cuda_threads) {
					std::string str = parent_gen_cuda_forward(n, cuda_threads);

					std::string outval = "out" + std::to_string(output_ref.offset);
					str += n.set_cuda_value_var(output_ref, outval);

					str += replace(R"(
					{
						value_t* input = $input;
						value_t* output = $output;
						for (size_t i = thread_index; i < $output_size; i += $threads) {
							output[i] = (value_t)1.0 / ((value_t)1.0 + exp(-input[i]));
						}
					}
					)",
					{
						{"$outval", outval},
						{"$threads", std::to_string(cuda_threads)},
						{"$input", n.gen_cuda_get_values(input_ref)},
						{"$output", n.gen_cuda_get_values(output_ref)},
						{"$output_size", std::to_string(output_ref.size)},
					});

					return str;
				};
				auto parent_gen_cuda_backward = n.gen_cuda_backward;
				n.gen_cuda_backward = [output_ref, input_gradients_ref, gradients_index, parent_gen_cuda_backward](nn& n, int cuda_threads) {

					std::string str = replace(R"(
					{
						value_t* input_gradients = $input_gradients;
						value_t* output = $output;
						$combine_gradients
						for (size_t i = 0; i != $input_gradients_ref_size; ++i) {
							input_gradients[i] = gradients[i] * ((value_t)1.0 - output[i]) * output[i];
						}
					}
					)",
					{
						{"$output", n.gen_cuda_get_values(output_ref)},
						{"$input_gradients", n.gen_cuda_get_values(input_gradients_ref)},
						{"$input_gradients_ref_size", std::to_string(input_gradients_ref.size)},
						{"$combine_gradients", n.gen_cuda_combine_gradients(gradients_index)}
					});

					return str + parent_gen_cuda_backward(n, cuda_threads);
				};
			});
			return { output_ref, gradients_index };
		}

		unit_ref make_tanh(unit_ref input_unit_ref) {
			vector_ref input_ref = input_unit_ref.output;
			vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
			vector_ref output_ref = new_vector_ref(input_ref.size);
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			construct_funcs.push_back([input_ref, input_gradients_ref, output_ref, gradients_index](nn& n) {
				auto parent_forward = n.forward;
				n.forward = [input_ref, output_ref, parent_forward](nn& n, value_t* in_weights) {
					parent_forward(n, in_weights);
					value_t* input = n.get_values(input_ref);
					value_t* output = n.get_values(output_ref);
					for (size_t i = 0; i < output_ref.size; ++i) {
						output[i] = std::tanh(input[i]);
					}
				};
				auto parent_backward = n.backward;
				n.backward = [input_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input_gradients = n.get_values(input_gradients_ref);
					value_t* output = n.get_values(output_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					for (size_t i = 0; i < input_gradients_ref.size; ++i) {
						input_gradients[i] = gradients[i] * ((value_t)1.0 - output[i] * output[i]);
					}
					parent_backward(n, in_weights, grad_in);
				};
				auto parent_gen_cuda_forward = n.gen_cuda_forward;
				n.gen_cuda_forward = [input_ref, output_ref, parent_gen_cuda_forward](nn& n, int cuda_threads) {
					std::string str = parent_gen_cuda_forward(n, cuda_threads);

					std::string outval = "out" + std::to_string(output_ref.offset);
					str += n.set_cuda_value_var(output_ref, outval);

					str += replace(R"(
					{
						value_t* input = $input;
						value_t* output = $output;
						for (size_t i = thread_index; i < $output_size; i += $threads) {
							output[i] = tanh(input[i]);
						}
					}
					)",
					{
						{"$outval", outval},
						{"$threads", std::to_string(cuda_threads)},
						{"$input", n.gen_cuda_get_values(input_ref)},
						{"$output", n.gen_cuda_get_values(output_ref)},
						{"$output_size", std::to_string(output_ref.size)},
					});

					return str;
				};
				auto parent_gen_cuda_backward = n.gen_cuda_backward;
				n.gen_cuda_backward = [output_ref, input_gradients_ref, gradients_index, parent_gen_cuda_backward](nn& n, int cuda_threads) {

					std::string str = replace(R"(
					{
						value_t* input_gradients = $input_gradients;
						value_t* output = $output;
						$combine_gradients
						for (size_t i = 0; i != $input_gradients_ref_size; ++i) {
							input_gradients[i] = gradients[i] * ((value_t)1.0 - output[i] * output[i]);
						}
					}
					)",
					{
						{"$output", n.gen_cuda_get_values(output_ref)},
						{"$input_gradients", n.gen_cuda_get_values(input_gradients_ref)},
						{"$input_gradients_ref_size", std::to_string(input_gradients_ref.size)},
						{"$combine_gradients", n.gen_cuda_combine_gradients(gradients_index)}
					});

					return str + parent_gen_cuda_backward(n, cuda_threads);
				};
			});
			return { output_ref, gradients_index };
		}

		unit_ref make_relu(unit_ref input_unit_ref) {
			vector_ref input_ref = input_unit_ref.output;
			vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
			vector_ref output_ref = new_vector_ref(input_ref.size);
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			construct_funcs.push_back([input_ref, input_gradients_ref, output_ref, gradients_index](nn& n) {
				auto parent_forward = n.forward;
				n.forward = [input_ref, output_ref, parent_forward](nn& n, value_t* in_weights) {
					parent_forward(n, in_weights);
					value_t* input = n.get_values(input_ref);
					value_t* output = n.get_values(output_ref);
					for (size_t i = 0; i < output_ref.size; ++i) {
						output[i] = input[i] < 0 ? 0 : input[i];
					}
				};
				auto parent_backward = n.backward;
				n.backward = [input_ref, input_gradients_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input = n.get_values(input_ref);
					value_t* input_gradients = n.get_values(input_gradients_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					for (size_t i = 0; i < input_gradients_ref.size; ++i) {
						input_gradients[i] = input[i] < 0 ? 0 : gradients[i];
					}
					parent_backward(n, in_weights, grad_in);
				};
				auto parent_gen_cuda_forward = n.gen_cuda_forward;
				n.gen_cuda_forward = [input_ref, output_ref, parent_gen_cuda_forward](nn& n, int cuda_threads) {
					std::string str = parent_gen_cuda_forward(n, cuda_threads);

					std::string outval = "out" + std::to_string(output_ref.offset);
					str += n.set_cuda_value_var(output_ref, outval);

					str += replace(R"(
					{
						value_t* input = $input;
						value_t* output = $output;
						for (size_t i = thread_index; i < $output_size; i += $threads) {
							output[i] = input[i] < 0 ? 0 : input[i];
						}
					}
					)",
					{
						{"$outval", outval},
						{"$threads", std::to_string(cuda_threads)},
						{"$input", n.gen_cuda_get_values(input_ref)},
						{"$output", n.gen_cuda_get_values(output_ref)},
						{"$output_size", std::to_string(output_ref.size)},
					});

					return str;
				};
				auto parent_gen_cuda_backward = n.gen_cuda_backward;
				n.gen_cuda_backward = [input_ref, input_gradients_ref, gradients_index, parent_gen_cuda_backward](nn& n, int cuda_threads) {

					std::string str = replace(R"(
					{
						value_t* input = $input;
						value_t* input_gradients = $input_gradients;
						$combine_gradients
						for (size_t i = 0; i != $input_size; ++i) {
							input_gradients[i] = input[i] < 0 ? 0 : gradients[i];
						}
					}
					)",
					{
						{"$input", n.gen_cuda_get_values(input_ref)},
						{"$input_gradients", n.gen_cuda_get_values(input_gradients_ref)},
						{"$input_size", std::to_string(input_ref.size)},
						{"$combine_gradients", n.gen_cuda_combine_gradients(gradients_index)}
					});

					return str + parent_gen_cuda_backward(n, cuda_threads);
				};
			});
			return { output_ref, gradients_index };
		}

		unit_ref make_select(size_t offset, size_t size, unit_ref input_unit_ref) {
			vector_ref output_ref = input_unit_ref.output.select(offset, size);
			gradients_index_ref gradients_index = select_gradients_index(input_unit_ref.gradients_index, offset, size);
			return { output_ref, gradients_index };
		}

		unit_ref make_add(unit_ref input_a_unit_ref, unit_ref input_b_unit_ref) {
			vector_ref input_a_ref = input_a_unit_ref.output;
			vector_ref input_a_gradients_ref = new_gradient(input_a_unit_ref.gradients_index);
			vector_ref input_b_ref = input_b_unit_ref.output;
			vector_ref input_b_gradients_ref = new_gradient(input_b_unit_ref.gradients_index);
			if (input_a_ref.size != input_b_ref.size) throw std::runtime_error("size mismatch");
			vector_ref output_ref = new_vector_ref(input_a_ref.size);
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			construct_funcs.push_back([input_a_ref, input_a_gradients_ref, input_b_ref, input_b_gradients_ref, output_ref, gradients_index](nn& n) {
				auto parent_forward = n.forward;
				n.forward = [input_a_ref, input_b_ref, output_ref, parent_forward](nn& n, value_t* in_weights) {
					parent_forward(n, in_weights);
					value_t* input_a = n.get_values(input_a_ref);
					value_t* input_b = n.get_values(input_b_ref);
					value_t* output = n.get_values(output_ref);
					for (size_t i = 0; i < output_ref.size; ++i) {
						output[i] = input_a[i] + input_b[i];
					}
				};
				auto parent_backward = n.backward;
				n.backward = [input_a_gradients_ref, input_b_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input_a_gradients = n.get_values(input_a_gradients_ref);
					value_t* input_b_gradients = n.get_values(input_b_gradients_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					for (size_t i = 0; i < gradients_index.size; ++i) {
						input_a_gradients[i] = gradients[i];
						input_b_gradients[i] = gradients[i];
					}
					parent_backward(n, in_weights, grad_in);
				};
				auto parent_gen_cuda_forward = n.gen_cuda_forward;
				n.gen_cuda_forward = [input_a_ref, input_b_ref, output_ref, parent_gen_cuda_forward](nn& n, int cuda_threads) {
					std::string str = parent_gen_cuda_forward(n, cuda_threads);

					str += replace(R"(
					{
						value_t* input_a = $input_a;
						value_t* input_b = $input_b;
						value_t* output = $output;
						for (size_t i = 0; i < $output_size; ++i) {
							output[i] = input_a[i] + input_b[i];
						}
					}
					)",
					{
						{"$input_a", n.gen_cuda_get_values(input_a_ref)},
						{"$input_b", n.gen_cuda_get_values(input_b_ref)},
						{"$output", n.gen_cuda_get_values(output_ref)},
						{"$output_size", std::to_string(output_ref.size)},
					});

					return str;
				};
				auto parent_gen_cuda_backward = n.gen_cuda_backward;
				n.gen_cuda_backward = [input_a_gradients_ref, input_b_gradients_ref, gradients_index, parent_gen_cuda_backward](nn& n, int cuda_threads) {

					std::string str = replace(R"(
					{
						value_t* input_a_gradients = $input_a_gradients;
						value_t* input_b_gradients = $input_b_gradients;
						$combine_gradients
						for (size_t i = 0; i != $gradients_index_size; ++i) {
							input_a_gradients[i] = gradients[i];
							input_b_gradients[i] = gradients[i];
						}
					}
					)",
					{
						{"$input_a_gradients", n.gen_cuda_get_values(input_a_gradients_ref)},
						{"$input_b_gradients", n.gen_cuda_get_values(input_b_gradients_ref)},
						{"$gradients_index_size", std::to_string(gradients_index.size)},
						{"$combine_gradients", n.gen_cuda_combine_gradients(gradients_index)}
					});

					return str + parent_gen_cuda_backward(n, cuda_threads);
				};
			});
			return { output_ref, gradients_index };
		}

		unit_ref make_mul(unit_ref input_a_unit_ref, unit_ref input_b_unit_ref) {
			vector_ref input_a_ref = input_a_unit_ref.output;
			vector_ref input_a_gradients_ref = new_gradient(input_a_unit_ref.gradients_index);
			vector_ref input_b_ref = input_b_unit_ref.output;
			vector_ref input_b_gradients_ref = new_gradient(input_b_unit_ref.gradients_index);
			if (input_a_ref.size != input_b_ref.size) throw std::runtime_error("size mismatch");
			vector_ref output_ref = new_vector_ref(input_a_ref.size);
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			construct_funcs.push_back([input_a_ref, input_a_gradients_ref, input_b_ref, input_b_gradients_ref, output_ref, gradients_index](nn& n) {
				auto parent_forward = n.forward;
				n.forward = [input_a_ref, input_b_ref, output_ref, parent_forward](nn& n, value_t* in_weights) {
					parent_forward(n, in_weights);
					value_t* input_a = n.get_values(input_a_ref);
					value_t* input_b = n.get_values(input_b_ref);
					value_t* output = n.get_values(output_ref);
					for (size_t i = 0; i < output_ref.size; ++i) {
						output[i] = input_a[i] * input_b[i];
					}
				};
				auto parent_backward = n.backward;
				n.backward = [input_a_ref, input_a_gradients_ref, input_b_ref, input_b_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input_a = n.get_values(input_a_ref);
					value_t* input_a_gradients = n.get_values(input_a_gradients_ref);
					value_t* input_b = n.get_values(input_b_ref);
					value_t* input_b_gradients = n.get_values(input_b_gradients_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					for (size_t i = 0; i < gradients_index.size; ++i) {
						input_a_gradients[i] = gradients[i] * input_b[i];
						input_b_gradients[i] = gradients[i] * input_a[i];
					}
					parent_backward(n, in_weights, grad_in);
				};
				n.gen_cuda_forward = [](int cuda_threads) {
					return std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cuda not implemented";
				};
				n.gen_cuda_backward = [](nn& n, int cuda_threads) {
					return std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cuda not implemented";
				};
			});
			return { output_ref, gradients_index };
		}

		unit_ref make_dropout(unit_ref input_unit_ref, value_t prob) {
			vector_ref input_ref = input_unit_ref.output;
			vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
			vector_ref output_ref = new_vector_ref(input_ref.size);
			vector_ref mask_ref = new_vector_ref(input_ref.size);
			gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			construct_funcs.push_back([prob, input_ref, input_gradients_ref, output_ref, mask_ref, gradients_index](nn& n) {
				auto parent_forward = n.forward;
				n.forward = [prob, input_ref, output_ref, mask_ref, parent_forward](nn& n, value_t* in_weights) {
					parent_forward(n, in_weights);
					value_t* input = n.get_values(input_ref);
					value_t* output = n.get_values(output_ref);
					value_t* mask = n.get_values(input_ref);
					if (!n.is_training && !n.is_evaluating) throw std::runtime_error("please call set_training() or set_evaluating()");
					if (n.is_training) {
						for (size_t i = 0; i < output_ref.size; ++i) {
							mask[i] = std::generate_canonical<value_t, 15>(n.rng_e) <= prob ? (value_t)0.0 : (value_t)1.0;
							output[i] = input[i] * mask[i];
						}
					} else {
						for (size_t i = 0; i < output_ref.size; ++i) {
							output[i] = input[i] * ((value_t)1.0 - prob);
						}
					}
				};
				auto parent_backward = n.backward;
				n.backward = [prob, input_ref, input_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
					value_t* input = n.get_values(input_ref);
					value_t* input_gradients = n.get_values(input_gradients_ref);
					value_t* output = n.get_values(output_ref);
					value_t* mask = n.get_values(input_ref);
					value_t* gradients = n.combine_gradients(gradients_index);
					if (n.is_training) {
						for (size_t i = 0; i < input_gradients_ref.size; ++i) {
							input_gradients[i] = gradients[i] * mask[i];
						}
					} else {
						for (size_t i = 0; i < input_gradients_ref.size; ++i) {
							input_gradients[i] = gradients[i] * ((value_t)1.0 - prob);
						}
					}
					parent_backward(n, in_weights, grad_in);
				};
				n.gen_cuda_forward = [](int cuda_threads) {
					return std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cuda not implemented";
				};
				n.gen_cuda_backward = [](nn& n, int cuda_threads) {
					return std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": cuda not implemented";
				};
			});
			return { output_ref, gradients_index };
		}

#ifdef TSCNN_CUDA
		CUdevice cuDevice;
		CUcontext context;
		CUmodule module;

		void set_device_context() {
			check_cu(cuCtxSetCurrent(context));
		}

		std::string cuda_src_override;

		CUfunction kernel_forward;
		CUfunction kernel_forward_backward;
		template<typename criterion_T>
		void make_cuda_kernels(criterion_T&& criterion) {
			check_cu(cuInit(0));
			check_cu(cuDeviceGet(&cuDevice, 0));
			check_cu(cuDevicePrimaryCtxRetain(&context, cuDevice));
			set_device_context();

			std::string src = gen_cuda_forward_kernel() + "\n\n";
			src += gen_cuda_forward_backward_kernel(criterion);

			if (!cuda_src_override.empty()) src = cuda_src_override;

			nvrtcProgram prog;
			//printf("compiling '%s'\n", src.c_str());
			check_nvrtc(nvrtcCreateProgram(&prog, src.c_str(), nullptr, 0, nullptr, nullptr));

			std::vector<std::string> ops;
			ops.push_back("--gpu-architecture=compute_60");
			ops.push_back("--use_fast_math");
			std::vector<const char*> opss;
			for (auto& v : ops) opss.push_back(v.data());
			auto err = nvrtcCompileProgram(prog, opss.size(), opss.data());

			printf("%s\n", src.c_str());

			if (err != NVRTC_SUCCESS) {
				printf("failed to compile '%s'\n", src.c_str());
				size_t log_size;
				check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size));
				std::vector<char> log;
				log.resize(log_size);
				check_nvrtc(nvrtcGetProgramLog(prog, log.data()));
				printf("%s\n", log.data());

				check_nvrtc(err);
			}

			size_t ptx_size = 0;
			check_nvrtc(nvrtcGetPTXSize(prog, &ptx_size));
			std::vector<char> ptx;
			ptx.resize(ptx_size);
			check_nvrtc(nvrtcGetPTX(prog, ptx.data()));

			check_nvrtc(nvrtcDestroyProgram(&prog));

			check_cu(cuModuleLoadDataEx(&module, ptx.data(), 0, 0, 0));
			check_cu(cuModuleGetFunction(&kernel_forward, module, "forward"));
			check_cu(cuModuleGetFunction(&kernel_forward_backward, module, "forward_backward"));

		}

		CUdeviceptr cuda_values = 0;

		CUdeviceptr get_cuda_values_for_batch(size_t n) {
			return cuda_values + sizeof(value_t) * values_batch_size * n;
		}

		CUdeviceptr get_cuda_values(vector_ref ref, size_t batch_n) {
			return get_cuda_values_for_batch(batch_n) + sizeof(value_t) * ref.offset;
		}

		void allocate_cuda_values() {
			check_cu(cuMemAlloc_v2(&cuda_values, sizeof(value_t) * values.size()));
		}

		void copy_to_cuda(value_t* src, vector_ref dst, size_t batch_n) {
			if (!cuda_values) throw std::runtime_error("please call allocate_cuda_values first");
			//printf("get_cuda_values(dst, batch_n) is %#x\n", get_cuda_values(dst, batch_n));
			check_cu(cuMemcpyHtoD_v2(get_cuda_values(dst, batch_n), src, sizeof(value_t) * dst.size));
		}

		void copy_to_cpu(vector_ref src, value_t* dst, size_t batch_n) {
			if (!cuda_values) throw std::runtime_error("please call allocate_cuda_values first");
			//printf("copy to cpu %#x from %#x\n", dst, get_cuda_values(src, batch_n));
			check_cu(cuMemcpyDtoH_v2(dst, get_cuda_values(src, batch_n), sizeof(value_t) * src.size));
		}

		void cuda_memset(vector_ref mem, value_t value, size_t batch_n) {
			if (sizeof(value_t) == 4) cuMemsetD32(get_cuda_values(mem, batch_n), (uint32_t&)value, mem.size);
			else if (sizeof(value_t) == 2) cuMemsetD16(get_cuda_values(mem, batch_n), (uint16_t&)value, mem.size);
			else cuMemsetD8(get_cuda_values(mem, batch_n), (uint8_t&)value, mem.size);
		}

		void cuda_forward(nn& n, vector_ref weights, size_t batch_size) {
			if (!n.cuda_values) throw std::runtime_error("please call allocate_cuda_values first");
			CUdeviceptr weights_ptr = get_cuda_values(weights, 0);
			std::array<void*, 2> args = {&cuda_values, &weights_ptr};
			check_cu(cuLaunchKernel(kernel_forward, batch_size, 1, 1, cuda_threads, 1, 1, 0, nullptr, args.data(), nullptr));
		}

		void cuda_forward_backward(nn& n, vector_ref weights, vector_ref grad, vector_ref criterion_input, int criterion_input_size, vector_ref criterion_target, vector_ref criterion_loss, vector_ref criterion_gradient, size_t batch_size) {
			if (!n.cuda_values) throw std::runtime_error("please call allocate_cuda_values first");
			CUdeviceptr weights_ptr = get_cuda_values(weights, 0);
			CUdeviceptr grad_ptr = get_cuda_values(grad, 0);
			CUdeviceptr criterion_input_ptr = get_cuda_values(criterion_input, 0);
			CUdeviceptr criterion_target_ptr = get_cuda_values(criterion_target, 0);
			CUdeviceptr criterion_loss_ptr = get_cuda_values(criterion_loss, 0);
			CUdeviceptr criterion_gradient_ptr = get_cuda_values(criterion_gradient, 0);
			std::array<void*, 8> args = {&cuda_values, &weights_ptr, &grad_ptr, &criterion_input_ptr, &criterion_input_size, &criterion_target_ptr, &criterion_loss_ptr, &criterion_gradient_ptr};
			check_cu(cuLaunchKernel(kernel_forward_backward, batch_size, 1, 1, cuda_threads, 1, 1, 0, nullptr, args.data(), nullptr));
		}

		void cuda_synchronize() {
			check_cu(cuCtxSynchronize());
		}
#endif

	};

	template<typename value_t = float>
	struct criterion_mse {
		void forward(size_t input_n, value_t* input, value_t* target, value_t* output) {
			value_t sum = 0.0;
			for (size_t i = 0; i < input_n; ++i) {
				value_t diff = target[i] - input[i];
				sum += diff*diff;
			}
			output[0] = sum / input_n;
		}

		void backward(size_t input_n, value_t* input, value_t* target, value_t* output) {
			value_t n = (value_t)2.0 / input_n;
			for (size_t i = 0; i < input_n; ++i) {
				output[i] = (input[i] - target[i]) * n;
			}
		}

	std::string gen_cuda_forward(nn<value_t>& n, int cuda_threads) {
			std::string str;

			str += n.cuda_check_sync();

			str += replace(R"(
			{
				if (thread_index == 0) {
					value_t* input = criterion_input;
					int input_size = criterion_input_size;
					value_t* target = criterion_target;
					value_t sum = 0.0;
					for (size_t i = 0; i < input_size; ++i) {
						value_t diff = target[i] - input[i];
						sum += diff*diff;
					}
					criterion_loss[batch_index] = sum / input_size;
				}
			}
			)",
			{
			});

			return str;
		}
		std::string gen_cuda_backward(nn<value_t>& n, int cuda_threads) {
			std::string str;

			str += n.cuda_check_sync();

			str += replace(R"(
			{
				value_t* input = criterion_input;
				int input_size = criterion_input_size;
				value_t* target = criterion_target;
				value_t* output = criterion_gradient;
				value_t n = (value_t)2.0 / input_size;
				for (size_t i = thread_index; i < input_size; i += $threads) {
					output[i] = (input[i] - target[i]) * n;
				}
			}
			)",
			{
				{"$threads", std::to_string(cuda_threads)},
			});

			n.cuda_set_sync();

			return str;
		}
	};

	template<typename value_t = float, typename allocator_T = std::allocator<value_t>>
	struct criterion_cross_entropy {
		std::vector<value_t, typename std::allocator_traits<allocator_T>::template rebind_alloc<value_t>> softmax_output;
		void forward(size_t input_n, value_t* input, value_t* target, value_t* output) {
			if (softmax_output.size() != input_n) softmax_output.resize(input_n);

			value_t highest = input[0];
			for (size_t i = 1; i < input_n; ++i) {
				if (input[i] > highest) highest = input[i];
			}
			value_t sum = 0.0;
			for (size_t i = 0; i < input_n; ++i) {
				sum += std::exp(input[i] - highest);
			}
			value_t log_sum = std::log(sum);
			for (size_t i = 0; i < input_n; ++i) {
				softmax_output[i] = input[i] - highest - log_sum;
			}

			output[0] = 0.0;
			for (size_t i = 0; i < input_n; ++i) {
				output[0] += target[i] * -softmax_output[i];
			}
		}

		void backward(size_t input_n, value_t* input, value_t* target, value_t* output) {
			for (size_t i = 0; i < input_n; ++i) {
				output[i] = (target[i] * (value_t)-1.0 - std::exp(softmax_output[i]) * (value_t)-1.0);
			}
		}
	};

	template<typename value_t = float, typename allocator_T = std::allocator<void>>
	struct rmsprop {
		value_t learning_rate = (value_t)0.001;
		value_t alpha = (value_t)0.9;
		value_t epsilon = (value_t)1e-8;
		value_t weight_decay = 0;
		value_t momentum = 0;

		std::vector<value_t, typename std::allocator_traits<allocator_T>::template rebind_alloc<value_t>> mean_squared;
		std::vector<value_t, typename std::allocator_traits<allocator_T>::template rebind_alloc<value_t>> current_momentum;

		void operator()(value_t* weights, value_t* grad, size_t n_grad) {
			if (mean_squared.size() != n_grad) mean_squared.resize(n_grad);
			if (current_momentum.size() != n_grad) current_momentum.resize(n_grad);
			value_t learning_rate = this->learning_rate;
			value_t alpha = this->alpha;
			value_t epsilon = this->epsilon;
			value_t weight_decay = this->weight_decay;
			value_t momentum = this->momentum;
			for (size_t i = 0; i < n_grad; ++i) {
				value_t w = weights[i];
				value_t& v = mean_squared[i];
				value_t g = grad[i] + w * weight_decay;
				v *= alpha;
				v += ((value_t)1.0 - alpha) * g*g;
				value_t m = std::sqrt(v) + epsilon;
				value_t adjust = g / m * learning_rate;

				value_t& mom = current_momentum[i];
				value_t prev_mom = mom;
				mom = (mom + adjust) * momentum;

				weights[i] = w - (adjust + prev_mom);
			}
		}

	};

#ifdef TSCNN_CUDA
#undef check_nvrtc
#undef check_cu
#endif

}
