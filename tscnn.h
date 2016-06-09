
#ifndef TSCNN_H
#define TSCNN_H

#include <memory>
#include <functional>
#include <vector>

namespace tscnn {

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

		template<typename>
		struct a_function;

		template<typename R, typename... args_T>
		class a_function<R(args_T...)> {
			std::function<R(args_T...)> f;
		public:
			a_function() : f() {};
			a_function(std::nullptr_t) : f(nullptr) {}
			template<typename F>
			a_function(F&& f) : f(std::allocator_arg, allocator_T(), f) {}
// 			template< class F >
// 			a_function& operator=(F&& f) {
// 				this->f = std::forward<F>(f);
// 				return *this;
// 			}
			template<typename... A>
			R operator()(A&&... args) const {
				return f(std::forward<A>(args)...);
			}
		};

		a_function<void(nn&, value_t*)> forward = [](nn&, value_t*) {};
		a_function<void(nn&, value_t*, value_t*)> backward = [](nn&, value_t*, value_t*) {};

		//a_vector<a_function<void(nn&)>> backward_construct_funcs;
		a_vector<a_function<void(nn&)>> construct_funcs;

		a_vector<value_t> values;
		size_t total_weights = 0;

		a_vector<std::pair<gradients_index_ref, a_vector<std::pair<vector_ref, gradients_index_ref>>>> gradients;
		a_vector<bool> gradient_increment_first;

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

		void construct() {
			// 		for (auto i = backward_construct_funcs.rbegin(); i != backward_construct_funcs.rend(); ++i) {
			// 			(*i)(*this);
			// 		}
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

		gradients_index_ref new_gradients_index(size_t size) {
			gradients.emplace_back();
			return gradients.back().first = { gradients.size() - 1, 0, size, size };
		}

		gradients_index_ref select_gradients_index(gradients_index_ref index, size_t offset, size_t size) {
			return { index.index, offset, size, index.size };
		}

		value_t* combine_gradients(gradients_index_ref index) {
			auto& grads = gradients[index.index].second;
			if (grads.empty()) xcept("missing gradients (some output is left unconnected ?)");
			size_t size = index.size;
			if (grads[0].first.size != size) xcept("base gradients size mismatch");
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
						if (std::isnan(src[i2])) xcept("combine src is nan");
						dst[offset + i2] += src[i2];
					}
				}
			}
			return dst;
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
					size_t input_size = input_ref.size;
					size_t output_size = output_ref.size;
					value_t* bias_w = in_weights + weights_offset;
					value_t* w = bias_w + output_size;
					//log("linear output: \n");
					// 				for (size_t oi = 0; oi < output_size; ++oi) {
					// 					//log("linear %d bias is %g\n", oi, *w);
					// 					output[oi] = *w++;
					// 				}
					for (size_t oi = 0; oi < output_size; ++oi) {
						//log("linear %d bias is %g\n", oi, *w);
						output[oi] = *bias_w++;
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
			vector_ref output_ref = input_unit_ref.output.select(offset, size);
			gradients_index_ref gradients_index = select_gradients_index(input_unit_ref.gradients_index, offset, size);
			return { output_ref, gradients_index };
			// 		vector_ref input_ref = input_unit_ref.output;
			// 		vector_ref input_gradients_ref = new_gradient(input_unit_ref.gradients_index);
			// 		vector_ref output_ref = new_vector_ref(size);
			// 		gradients_index_ref gradients_index = new_gradients_index(output_ref.size);
			// 		log("made select gradient with size %d\n", gradients_index.size);
			// 		construct_funcs.push_back([offset, input_ref, input_gradients_ref, output_ref, gradients_index](nn& n) {
			// 			auto parent_forward = n.forward;
			// 			n.forward = [offset, input_ref, output_ref, parent_forward](nn& n, value_t* in_weights) {
			// 				parent_forward(n, in_weights);
			// 				value_t* input = n.get_values(input_ref);
			// 				value_t* output = n.get_values(output_ref);
			// 				for (size_t i = 0; i < output_ref.size; ++i) {
			// 					output[i] = input[offset + i];
			// 				}
			// 			};
			// 			auto parent_backward = n.backward;
			// 			n.backward = [input_gradients_ref, output_ref, gradients_index, parent_backward](nn& n, value_t* in_weights, value_t* grad_in) {
			// 				value_t* input_gradients = n.get_values(input_gradients_ref);
			// 				value_t* output = n.get_values(output_ref);
			// 				value_t* gradients = n.combine_gradients(gradients_index);
			// 				for (size_t i = 0; i < output_ref.size; ++i) {
			// 					input_gradients[i] = gradients[i];
			// 				}
			// 				parent_backward(n, in_weights, grad_in);
			// 			};
			// 		});
			// 		return { output_ref, gradients_index };
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
		value_t alpha = (value_t)0.99;
		value_t epsilon = (value_t)1e-8;

		std::vector<value_t, typename std::allocator_traits<allocator_T>::template rebind_alloc<value_t>> momentum_squared;

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

}

#endif
