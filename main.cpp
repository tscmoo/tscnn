#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <cstring>

#include <array>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <functional>
#include <numeric>
#include <memory>
#include <random>
#include <chrono>
#include <thread>
#include <initializer_list>
#include <mutex>
#include <algorithm>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#endif

#undef min
#undef max
#undef near
#undef far

//#include "tsc/intrusive_list.h"
//#include "tsc/alloc.h"
template<typename T>
using alloc = std::allocator<T>;
#include "tsc/alloc_containers.h"

#include "tsc/strf.h"

constexpr bool test_mode = true;

int current_frame;

struct simple_logger {
	std::mutex mut;
	tsc::a_string str, str2;
	bool newline = true;
	FILE*f = nullptr;
	simple_logger() {
		if (test_mode) f = fopen("log.txt", "w");
	}
	template<typename...T>
	void operator()(const char*fmt, T&&...args) {
		std::lock_guard<std::mutex> lock(mut);
		try {
			tsc::strf::format(str, fmt, std::forward<T>(args)...);
		} catch (const std::exception&) {
			str = fmt;
		}
		if (newline) tsc::strf::format(str2, "%5d: %s", current_frame, str);
		const char*out_str = newline ? str2.c_str() : str.c_str();
		newline = strchr(out_str, '\n') ? true : false;
		if (f) {
			fputs(out_str, f);
			fflush(f);
		}
		fputs(out_str, stdout);
	}
};
simple_logger logger;


enum {
	log_level_all,
	log_level_debug,
	log_level_info
};

int current_log_level = test_mode ? log_level_all : log_level_info;
//int current_log_level = log_level_info;

template<typename...T>
void log(int level, const char*fmt, T&&...args) {
	if (current_log_level <= level) logger(fmt, std::forward<T>(args)...);
}

template<typename...T>
void log(const char*fmt, T&&...args) {
	log(log_level_debug, fmt, std::forward<T>(args)...);
}

struct xcept_t {
	tsc::a_string str1, str2;
	int n;
	xcept_t() {
		str1.reserve(0x100);
		str2.reserve(0x100);
		n = 0;
	}
	template<typename...T>
	void operator()(const char*fmt, T&&...args) {
		try {
			auto&str = ++n % 2 ? str1 : str2;
			tsc::strf::format(str, fmt, std::forward<T>(args)...);
			log(log_level_info, "about to throw exception %s\n", str);
			//#ifdef _DEBUG
			//DebugBreak();
			//#endif
			throw (const char*)str.c_str();
		} catch (const std::bad_alloc&) {
			throw (const char*)fmt;
		}
	}
};
xcept_t xcept;

tsc::a_string format_string;
template<typename...T>
const char*format(const char*fmt, T&&...args) {
	return tsc::strf::format(format_string, fmt, std::forward<T>(args)...);
}

#include "tsc/userthreads.h"

#include "tsc/high_resolution_timer.h"
#include "tsc/rng.h"
#include "tsc/bitset.h"

using tsc::rng;
using tsc::a_string;
using tsc::a_vector;
using tsc::a_deque;
using tsc::a_list;
using tsc::a_set;
using tsc::a_multiset;
using tsc::a_map;
using tsc::a_multimap;
using tsc::a_unordered_set;
using tsc::a_unordered_multiset;
using tsc::a_unordered_map;
using tsc::a_unordered_multimap;

#include "tsc/json.h"

struct default_link_f {
	template<typename T>
	auto* operator()(T*ptr) {
		return (std::pair<T*, T*>*)&ptr->link;
	}
};

const double PI = 3.1415926535897932384626433;

using value_t = float;


struct nn {

	using apply_func = std::function<value_t*(nn&)>;

	std::function<void(nn&, value_t*)> forward = [](nn&, value_t*) {};
	std::function<void(nn&, value_t*, value_t*)> backward = [](nn&, value_t*, value_t*) {};

	a_vector<std::function<void(nn&)>> backward_construct_funcs;
	a_vector<std::function<void(nn&)>> construct_funcs;

	a_vector<value_t> values;
	size_t total_weights = 0;

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

	vector_ref new_vector_ref(size_t size) {
		if ((uintptr_t)(values.data() + values.size()) % 4 != 0) values.resize(((uintptr_t)(values.data() + values.size()) & -4) + 4 - (uintptr_t)values.data());
		size_t offset = values.size();
		values.resize(offset + size);
		return { offset, size };
	}

	value_t* get_values(vector_ref ref) {
		return values.data() + ref.offset;
	}

	void construct() {
		for (auto i = backward_construct_funcs.rbegin(); i != backward_construct_funcs.rend(); ++i) {
			(*i)(*this);
		}
		for (auto& v : construct_funcs) {
			v(*this);
		}
	}

	a_vector<a_vector<vector_ref>> gradients;

	gradients_index_ref new_gradients_index(size_t size) {
		gradients.emplace_back();
		return { gradients.size() - 1, 0, size, size };
	}

	gradients_index_ref select_gradients_index(gradients_index_ref index, size_t offset, size_t size) {
		return { index.index, offset, size, index.size };
	}

	value_t* combine_gradients(gradients_index_ref index) {
		auto& grads = gradients[index.index];
		if (grads.empty()) xcept("missing gradients");
		size_t size = index.size;
		if (grads[0].size != size) xcept("gradients size mismatch");
		value_t* dst = get_values(grads[0]);
		for (size_t i = 1; i < grads.size(); ++i) {
			if (grads[i].size != size) xcept("gradients size mismatch");
			value_t* src = get_values(grads[i]);
			for (size_t i2 = 0; i2 < size; ++i2) {
				dst[i2] += src[i2];
			}
		}
		return dst;
	}

	vector_ref new_gradient(gradients_index_ref index) {
		vector_ref r = new_vector_ref(index.allocation_size);
		gradients[index.index].push_back(r);
		r.offset += index.offset;
		r.size = index.size;
		return r;
	}

	struct unit_ref {
		vector_ref output;
		gradients_index_ref gradients_index;
	};

	a_vector<unit_ref> inputs;
	a_vector<unit_ref> outputs;

	double learning_rate = 0.01;

	unit_ref make_input(size_t out) {
		gradients_index_ref gradients_index = new_gradients_index(out);
		construct_funcs.push_back([gradients_index](nn& n) {
			auto parent_backward = n.backward;
			n.backward = [gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
				n.combine_gradients(gradients_index);
				parent_backward(n, in_weights, grad_in);
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
					if (std::isnan(gradients[i])) xcept("output gradient is nan");
					input_gradients[i] = gradients[i];
				}
				parent_backward(n, in_weights, grad_in);
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
		construct_funcs.push_back([input_ref, input_gradients_ref, output_ref, gradients_index, weights_offset](nn& n) {
			auto parent_forward = n.forward;
			n.forward = [input_ref, output_ref, weights_offset, parent_forward](nn& n, value_t* in_weights) {
				parent_forward(n, in_weights);
				value_t* input = n.get_values(input_ref);
				value_t* output = n.get_values(output_ref);
				value_t* w = in_weights + weights_offset;
				size_t input_size = input_ref.size;
				size_t output_size = output_ref.size;
				//log("linear output: \n");
				for (size_t oi = 0; oi < output_size; ++oi) {
					output[oi] = *w++;
					for (size_t ii = 0; ii < input_size; ++ii) {
						output[oi] += input[ii] * *w++;
					}
					//log("%g\n", output[oi]);
				}
			};
			auto parent_backward = n.backward;
			n.backward = [input_ref, input_gradients_ref, output_ref, gradients_index, weights_offset, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
				value_t* input = n.get_values(input_ref);
				value_t* input_gradients = n.get_values(input_gradients_ref);
				value_t* gradients = n.combine_gradients(gradients_index);
				value_t* w = in_weights + weights_offset;
				size_t input_size = input_ref.size;
				size_t output_size = output_ref.size;
				for (size_t ii = 0; ii < input_size; ++ii) {
					input_gradients[ii] = 0;
				}
				//log("output_size %d, input_size %d\n", output_size, input_size);
				for (size_t oi = 0; oi < output_size; ++oi) {
					++w;
					for (size_t ii = 0; ii < input_size; ++ii) {
						//log("ii %d %g += oi %d %g * %g\n", ii, input_gradients[ii], oi, gradients[oi], *w);
						input_gradients[ii] += gradients[oi] * *w++;
					}
				}
				// 				for (size_t ii = 0; ii < input_size; ++ii) {
				// 					input_gradients[ii] = 0;
				// 					value_t* column = w;
				// 					for (size_t oi = 0; oi < output_size; ++oi) {
				// 						input_gradients[ii] += gradients[oi] * *w;
				// 						w += input_size;
				// 					}
				// 					w = column + 1;
				// 				}
				value_t* g = grad_in + weights_offset;
				for (size_t oi = 0; oi < output_size; ++oi) {
					*g++ += gradients[oi];
					for (size_t ii = 0; ii < input_size; ++ii) {
						*g++ += gradients[oi] * input[ii];
					}
				}
				parent_backward(n, in_weights, grad_in);
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
		});
		return { output_ref, gradients_index };
	}

	unit_ref make_select(size_t offset, size_t size, unit_ref input_unit_ref) {
		// 		vector_ref output_ref = input_unit_ref.output.select(offset, size);
		// 		gradients_index_ref gradients_index = select_gradients_index(input_unit_ref.gradients_index, offset, size);
		// 		return { output_ref, gradients_index };
		vector_ref input_ref = input_unit_ref.output;
		vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
		vector_ref output_ref = new_vector_ref(size);
		gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
		construct_funcs.push_back([offset, input_ref, input_gradients_ref, output_ref, gradients_index](nn& n) {
			auto parent_forward = n.forward;
			n.forward = [offset, input_ref, output_ref, parent_forward](nn& n, value_t* in_weights) {
				parent_forward(n, in_weights);
				value_t* input = n.get_values(input_ref);
				value_t* output = n.get_values(output_ref);
				for (size_t i = 0; i < output_ref.size; ++i) {
					output[i] = input[offset + i];
				}
			};
			auto parent_backward = n.backward;
			n.backward = [offset, input_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
				value_t* input_gradients = n.get_values(input_gradients_ref);
				value_t* output = n.get_values(output_ref);
				value_t* gradients = n.combine_gradients(gradients_index);
				for (size_t i = 0; i < output_ref.size; ++i) {
					input_gradients[offset + i] = gradients[i];
				}
				parent_backward(n, in_weights, grad_in);
			};
		});
		return { output_ref, gradients_index };
	}

	unit_ref make_add(unit_ref input_a_unit_ref, unit_ref input_b_unit_ref) {
		vector_ref input_a_ref = input_a_unit_ref.output;
		vector_ref input_a_gradients_ref = new_gradient(input_a_unit_ref.gradients_index);
		vector_ref input_b_ref = input_b_unit_ref.output;
		vector_ref input_b_gradients_ref = new_gradient(input_b_unit_ref.gradients_index);
		if (input_a_ref.size != input_b_ref.size) xcept("size mismatch");
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
		});
		return { output_ref, gradients_index };
	}

	unit_ref make_mul(unit_ref input_a_unit_ref, unit_ref input_b_unit_ref) {
		vector_ref input_a_ref = input_a_unit_ref.output;
		vector_ref input_a_gradients_ref = new_gradient(input_a_unit_ref.gradients_index);
		vector_ref input_b_ref = input_b_unit_ref.output;
		vector_ref input_b_gradients_ref = new_gradient(input_b_unit_ref.gradients_index);
		if (input_a_ref.size != input_b_ref.size) xcept("size mismatch");
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
		});
		return { output_ref, gradients_index };
	}

};

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
};

struct rmsprop {
	value_t learning_rate = (value_t)0.001;
	value_t alpha = (value_t)0.99;
	value_t epsilon = (value_t)1e-8;

	a_vector<value_t> momentum_squared;

	void operator()(value_t* weights, value_t* grad, size_t n_grad) {
		if (momentum_squared.empty()) momentum_squared.resize(n_grad);
		value_t learning_rate = this->learning_rate;
		value_t alpha = this->alpha;
		value_t epsilon = this->epsilon;
		for (size_t i = 0; i < n_grad; ++i) {
			//log("%g\n", grad[i]);
			value_t& v = momentum_squared[i];
			value_t g = grad[i];
			v *= alpha;
			v += ((value_t)1.0 - alpha) * g*g;
			value_t m = std::sqrt(v) + epsilon;
			weights[i] -= g / m * learning_rate;
			//log("%g\n", weights[i]);
		}
	}

};

#include <cfenv>

int main() {

	FILE* f = fopen("input.txt", "rb");
	if (!f) xcept("failed to open input.txt");
	a_vector<char> input_data;
	fseek(f, 0, SEEK_END);
	input_data.resize(ftell(f));
	fseek(f, 0, SEEK_SET);
	fread(input_data.data(), input_data.size(), 1, f);
	fclose(f);

	a_unordered_set<char> characters;
	a_unordered_map<char, size_t> char_to_index;
	a_vector<char> index_to_char;
	for (char c : input_data) {
		if (characters.insert(c).second) {
			index_to_char.push_back(c);
			char_to_index[c] = index_to_char.size() - 1;
		}
	}

	size_t inputs = characters.size();
	size_t outputs = characters.size();
	const size_t state_size = 128;

	a_vector<size_t> forget_gate_offsets;

	nn a;

	auto nn_all_inputs = a.make_input(inputs + state_size * 2);

	forget_gate_offsets.push_back(a.total_weights);
	//auto nn_processed_input = a.make_linear(state_size * 4, a.make_select(0, inputs + state_size, nn_all_inputs));
	auto nn_processed_input_a = a.make_linear(state_size * 4, a.make_select(0, inputs, nn_all_inputs));
	auto nn_processed_input_b = a.make_linear(state_size * 4, a.make_select(inputs, state_size, nn_all_inputs));
	auto nn_processed_input = a.make_add(nn_processed_input_a, nn_processed_input_b);

	auto nn_cell_state_input = a.make_select(inputs + state_size, state_size, nn_all_inputs);

	auto nn_forget_gate = a.make_sigmoid(a.make_select(0, state_size, nn_processed_input));
	auto nn_in_gate = a.make_sigmoid(a.make_select(state_size, state_size, nn_processed_input));
	auto nn_in_scale_gate = a.make_tanh(a.make_select(state_size * 2, state_size, nn_processed_input));
	auto nn_out_gate = a.make_sigmoid(a.make_select(state_size * 3, state_size, nn_processed_input));

	auto nn_cell_state_post_forget = a.make_mul(nn_cell_state_input, nn_forget_gate);

	auto nn_add_cell_state = a.make_mul(nn_in_gate, nn_in_scale_gate);

	auto nn_cell_state_output = a.make_add(nn_cell_state_post_forget, nn_add_cell_state);

	auto nn_hidden_state_output = a.make_mul(nn_out_gate, a.make_tanh(nn_cell_state_output));

	auto nn_a_output = a.make_linear(outputs, nn_hidden_state_output);
	//auto nn_a_output = a.make_linear(outputs, nn_processed_input);

	// 	auto xnn_processed_input_a = a.make_linear(2, a.make_select(0, inputs, nn_all_inputs));
	// 	auto xnn_processed_input_b = a.make_linear(2, a.make_select(inputs, state_size, nn_all_inputs));
	// 	auto xnn_processed_input = a.make_add(xnn_processed_input_a, xnn_processed_input_b);
	// 	auto nn_a_output = xnn_processed_input;

	a.make_output(nn_a_output);
	a.make_output(nn_hidden_state_output);
	a.make_output(nn_cell_state_output);

	auto all_inputs = a.inputs[0];

	auto a_output = a.outputs[0];
	auto hidden_state_output = a.outputs[1];
	auto cell_state_output = a.outputs[2];

	auto a_output_gradients = a.new_gradient(a_output.gradients_index);
	auto hidden_state_output_gradients = a.new_gradient(hidden_state_output.gradients_index);
	auto cell_state_output_gradients = a.new_gradient(cell_state_output.gradients_index);

	a.construct();

	a.learning_rate = 0.001;

	log("a_output has %d gradients\n", a.gradients[a_output.gradients_index.index].size());
	log("hidden_state_output.gradients_index.index is %d\n", hidden_state_output.gradients_index.index);
	log("hidden has %d gradients\n", a.gradients[hidden_state_output.gradients_index.index].size());
	log("cell has %d gradients\n", a.gradients[cell_state_output.gradients_index.index].size());


	log("real hidden has %d gradients\n", a.gradients[nn_hidden_state_output.gradients_index.index].size());
	log("real cell has %d gradients\n", a.gradients[nn_cell_state_output.gradients_index.index].size());

	//return 0;

	criterion_mse criterion;

	// 	value_t* input = a.get_values(all_inputs.output);
	// 	value_t* output = a.get_values(a_output.output);
	// 
	// 	for (size_t i = 0; i < all_inputs.output.size; ++i) {
	// 		input[i] = 0.0;
	// 	}
	// 
	// 	input[0] = 0.25;
	// 	input[1] = 1.0;
	// 	input[2] = 1.0;

	a_vector<value_t> weights(a.total_weights);
	for (auto& v : weights) {
		v = -(value_t)0.1 + tsc::rng((value_t)0.2);
	}

	if (true) {
		// Initialize forget gate biases to 1
		value_t* w = weights.data();
		for (size_t oi = 0; oi < state_size; ++oi) {
			*w++ = 1.0;
			for (size_t ii = 0; ii < inputs + state_size; ++ii) {
				++w;
			}
		}
	}

	// 	for (auto& v : weights) {
	// 		v = 1.0;
	// 		v = 0.99;
	// 	}

	log("%d weights\n", weights.size());

	size_t seq_length = 50;

	struct net_info {
		nn n;

		value_t* input;
		value_t* output;

		value_t* hidden_state_output_values;
		value_t* cell_state_output_values;

		value_t* input_gradients;
		value_t* output_gradients;
		value_t* hidden_state_gradients;
		value_t* cell_state_gradients;
	};

	//for (auto& v : a.values) v = std::nan("");
	for (auto& v : a.values) v = std::numeric_limits<value_t>::signaling_NaN();

	a_vector<net_info> nets(seq_length);
	for (auto& v : nets) {
		v.n = a;

		v.input = v.n.get_values(all_inputs.output);
		v.output = v.n.get_values(a_output.output);

		v.hidden_state_output_values = v.n.get_values(hidden_state_output.output);
		v.cell_state_output_values = v.n.get_values(cell_state_output.output);
		for (size_t i = 0; i < state_size; ++i) {
			v.hidden_state_output_values[i] = 0.0;
			v.cell_state_output_values[i] = 0.0;
		}

		v.input_gradients = v.n.get_values(v.n.gradients[all_inputs.gradients_index.index][0]);
		v.output_gradients = v.n.get_values(a_output_gradients);
		v.hidden_state_gradients = v.n.get_values(hidden_state_output_gradients);
		v.cell_state_gradients = v.n.get_values(cell_state_output_gradients);
		for (size_t i = 0; i < a_output_gradients.size; ++i) {
			v.output_gradients[i] = 0.0;
		}
		for (size_t i = 0; i < state_size; ++i) {
			v.hidden_state_gradients[i] = 0.0;
			v.cell_state_gradients[i] = 0.0;
		}
	}

	a_vector<value_t> target_output(a_output.output.size);
	//a_vector<value_t> output_grad(a_output.output.size);

	//feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

	rmsprop opt;
	double avg = 0.0;

	size_t char_index = 0;

	for (int iteration = 0; iteration < 100000; ++iteration) {

		int a = 0;

		tsc::high_resolution_timer ht;

		a_vector<value_t> grad(weights.size());

		if (char_index + seq_length + 1 >= input_data.size()) char_index = 0;

		value_t loss = 0.0;
		for (size_t n = 0; n < seq_length; ++n) {

			auto& cur_net = nets[n];

			for (auto& v : cur_net.n.values) v = 0.0;

			char c = input_data[char_index];
			char nc = input_data[char_index + 1];
			++char_index;

			size_t c_index = char_to_index[c];
			size_t nc_index = char_to_index[nc];

			for (size_t i = 0; i < characters.size(); ++i) {
				cur_net.input[i] = c_index == i ? (value_t)1.0 : (value_t)0.0;
			}

			//cur_net.input[inputs + 0] = 1.0;
			//cur_net.input[inputs + state_size + 0] = 1.0;

			if (n != 0) {
				auto& prev_net = nets[n - 1];
				for (size_t i = 0; i < state_size; ++i) {
					cur_net.input[inputs + i] = prev_net.hidden_state_output_values[i];
					cur_net.input[inputs + state_size + i] = prev_net.cell_state_output_values[i];
				}
			} else {
				for (size_t i = 0; i < state_size; ++i) {
					cur_net.input[inputs + i] = 0.0;
					cur_net.input[inputs + state_size + i] = 0.0;
				}
			}

			cur_net.n.forward(cur_net.n, weights.data());

			//if (iteration >= 10000) {
			if (false) {
				log("input ");
				for (size_t i = 0; i < inputs + state_size * 2; ++i) {
					log(" %g", cur_net.input[i]);
				}
				log("\n");
				log("output ");
				for (size_t i = 0; i < outputs; ++i) {
					log(" %g", cur_net.output[i]);
				}
				log("\n");
				log("h ");
				for (size_t i = 0; i < state_size; ++i) {
					log(" %g", cur_net.hidden_state_output_values[i]);
				}
				log("\n");
				log("c ");
				for (size_t i = 0; i < state_size; ++i) {
					log(" %g", cur_net.cell_state_output_values[i]);
				}
				log("\n");

				// 				log("output of nn_out_gate: \n");
				// 				for (size_t i = 0; i < nn_out_gate.output.size; ++i) {
				// 					log(" %g", cur_net.n.get_values(nn_out_gate.output)[i]);
				// 				}
				// 				log("\n");
			}

			if (true) {
				value_t best_value = -std::numeric_limits<value_t>::infinity();
				size_t best_index = 0;
				for (size_t i = 0; i < outputs; ++i) {
					value_t v = cur_net.output[i];
					if (v > best_value) {
						best_value = v;
						best_index = i;
					}
				}
				log("%c", index_to_char[best_index]);
			}

			//xcept("stop");

			for (size_t i = 0; i < outputs; ++i) {
				if (std::isnan(cur_net.output[i])) xcept("nan output");
			}
			for (size_t i = 0; i < state_size; ++i) {
				if (std::isnan(cur_net.hidden_state_output_values[i])) xcept("nan output");
				if (std::isnan(cur_net.cell_state_output_values[i])) xcept("nan output");
			}

			for (size_t i = 0; i < characters.size(); ++i) {
				target_output[i] = nc_index == i ? (value_t)1.0 : (value_t)0.0;
			}

			std::array<value_t, 1> loss_arr;

			criterion.forward(a_output.output.size, cur_net.output, target_output.data(), loss_arr.data());

			loss += loss_arr[0];

			criterion.backward(a_output.output.size, cur_net.output, target_output.data(), cur_net.output_gradients);

			// 			log("criterion backward - ");
			// 			for (size_t i = 0; i < a_output.output.size; ++i) {
			// 				log(" %g", cur_net.output_gradients[i]);
			// 			}
			// 			log("\n");

			// 			for (size_t i = 0; i < a_output.output.size; ++i) {
			// 				rnn_output_gradients[n][i] = output_grad[i];
			// 			}

		}

		// 		for (size_t i = 0; i < state_size; ++i) {
		// 			hidden_state_gradients[i] = 0.0;
		// 			cell_state_gradients[i] = 0.0;
		// 		}

		for (size_t n = seq_length; n;) {
			--n;

			auto& cur_net = nets[n];

			if (n != seq_length - 1) {
				auto& next_net = nets[n + 1];

				for (size_t i = 0; i < state_size; ++i) {
					cur_net.hidden_state_gradients[i] = next_net.input_gradients[inputs + i];
					cur_net.cell_state_gradients[i] = next_net.input_gradients[inputs + state_size + i];
				}
			}

			// 			for (size_t i = 0; i < inputs; ++i) {
			// 				cur_net.output_gradients[i] = 0.0;
			// 			}

			// 			log("n %d\n", n);
			// 			log("output gradients - ");
			// 			for (size_t i = 0; i < inputs; ++i) {
			// 				log(" %g", cur_net.output_gradients[i]);
			// 			}
			// 			log("\n");
			// 			log("hidden state gradients - ");
			// 			for (size_t i = 0; i < state_size; ++i) {
			// 				log(" %g", cur_net.hidden_state_gradients[i]);
			// 			}
			// 			log("\n");
			// 			log("cell state gradients - ");
			// 			for (size_t i = 0; i < state_size; ++i) {
			// 				log(" %g", cur_net.cell_state_gradients[i]);
			// 			}
			// 			log("\n");


			cur_net.n.backward(cur_net.n, weights.data(), grad.data());

			// 			auto debug_grad = [&](nn::unit_ref u) {
			// 				log("a_output gradients (%d) -\n", cur_net.n.gradients[u.gradients_index.index].size());
			// 				for (size_t i = 0; i < u.gradients_index.size; ++i) {
			// 					log(" %g", cur_net.n.get_values(cur_net.n.gradients[u.gradients_index.index][0])[u.gradients_index.offset + i]);
			// 				}
			// 				log("\n");
			// 				for (size_t gi = 0; gi < cur_net.n.gradients[u.gradients_index.index].size(); ++gi) {
			// 					log("grad %d - ", gi);
			// 					for (size_t i = 0; i < u.gradients_index.size; ++i) {
			// 						log(" %g", cur_net.n.get_values(cur_net.n.gradients[u.gradients_index.index][gi])[u.gradients_index.offset + i]);
			// 					}
			// 					log("\n");
			// 				}
			// 			};
			// 			debug_grad(a_output);
			// 
			// 			log("gradients - \n");
			// 			for (size_t i = 0; i < inputs; ++i) {
			// 				log(" %g", cur_net.input_gradients[i]);
			// 			}
			// 			log("\n");
			// 			for (size_t i = 0; i < state_size; ++i) {
			// 				log(" %g", cur_net.input_gradients[inputs + i]);
			// 			}
			// 			log("\n");
			// 			for (size_t i = 0; i < state_size; ++i) {
			// 				log(" %g", cur_net.input_gradients[inputs + state_size + i]);
			// 			}
			// 			log("\n");
		}

		if (true) {
			auto& first_net = nets[0];
			auto& last_net = nets[seq_length - 1];
			for (size_t i = 0; i < state_size; ++i) {
				first_net.input[inputs + i] = last_net.hidden_state_output_values[i];
				first_net.input[inputs + state_size + i] = last_net.cell_state_output_values[i];
			}
		}

		//xcept("stop");

		// 			// 			input[0] = 1.0;
		// 			// 			input[1] = 0.0;
		// 
		// 			a.forward(a, weights.data());
		// 
		// 			//log("input %g %g\n", input[0], input[1]);
		// 			//log("output %g\n", output[0]);
		// 
		// 			std::array<value_t, 1> target_output;
		// 			//target_output[0] = va;
		// 			target_output[0] = va || vb ? 1.0 : 0.0;
		// 			//target_output[0] = va ^ vb ? 1.0 : 0.0;
		// 			//target_output[0] = va & vb ? 1.0 : 0.0;
		// 
		// 			std::array<value_t, 1> loss_arr;
		// 
		// 			criterion.forward(a_output.output.size, output, target_output.data(), loss_arr.data());
		// 
		// 			loss += loss_arr[0];
		// 			//log("loss %g\n", loss_arr[0]);
		// 
		// 			std::array<value_t, 1> output_grad;
		// 
		// 			criterion.backward(a_output.output.size, output, target_output.data(), output_grad.data());
		// 
		// 			//log("output grad %g\n", output_grad[0]);
		// 
		// 			a.backward(a, weights.data(), output_grad.data(), grad.data());
		// 
		// 			// 			log("weight gradients: \n");
		// 			// 			for (auto& v : grad) {
		// 			// 				log(" %g\n", v);
		// 			// 			}
		// 		}

		loss /= seq_length;

		//if (loss < 0.25) xcept("%d iterations\n", iteration);

		// 		for (auto& v : grad) {
		// 			v /= batch_n;
		// 		}

		for (size_t i = 0; i < weights.size(); ++i) {
			value_t& g = grad[i];
			if (g < -5.0) g = -5.0;
			if (g > 5.0) g = 5.0;
		}

		opt(weights.data(), grad.data(), grad.size());

		double t = ht.elapsed() * 1000;

		avg += t;

		//log("took %gms\n", );
		log("avg %gms\n", avg / (iteration + 1));

		// 		for (size_t i = 0; i < weights.size(); ++i) {
		// 			//log("gradient %d %g\n", i, grad[i]);
		// 			double g = grad[i];
		// 			//if (g < -100 || g > 100) xcept("g is %g\n", g);
		// 			if (g < -1.0) g = -1.0;
		// 			if (g > 1.0) g = 1.0;
		// 			weights[i] -= g * 1.0;
		// 			//log("weight %d -> %g\n", i, weights[i]);
		// 			if (std::isnan(weights[i])) {
		// 				log("gradient %d %g\n", i, grad[i]);
		// 				xcept("weight %d is nan\n", i);
		// 			}
		// 
		// 			//if (tsc::rng(1.0) < 1.0 / 64) {
		// 			//	weights[i] -= grad[i] * tsc::rng(1.0);
		// 			//}
		// 		}

		log("loss %g\n", loss);

		//if (iteration == 1000) break;

		// 		input[0] = 0.0;
		// 		input[1] = 0.0;
		// 
		// 		a.forward(a, weights.data());
		// 
		// 		log("hidden_state_output is \n");
		// 		for (size_t i = 0; i < hidden_state_output.output.size; ++i) {
		// 			log(" %g\n", a.get_values(hidden_state_output.output)[i]);
		// 		}
		// 
		// 		log("input %g %g\n", input[0], input[1]);
		// 		log("output %g\n", output[0]);
		// 
		// 		input[0] = 1.0;
		// 		input[1] = 0.0;
		// 
		// 		a.forward(a, weights.data());
		// 
		// 		log("input %g %g\n", input[0], input[1]);
		// 		log("output %g\n", output[0]);
		// 
		// 		input[0] = 0.0;
		// 		input[1] = 1.0;
		// 
		// 		a.forward(a, weights.data());
		// 
		// 		log("input %g %g\n", input[0], input[1]);
		// 		log("output %g\n", output[0]);
		// 
		// 		input[0] = 1.0;
		// 		input[1] = 1.0;
		// 
		// 		a.forward(a, weights.data());
		// 
		// 		log("input %g %g\n", input[0], input[1]);
		// 		log("output %g\n", output[0]);
	}

	return 0;
}


