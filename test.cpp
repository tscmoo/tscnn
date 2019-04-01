#include <exception>
#include <string>
#include <vector>
#include <array>
#include <numeric>
#include <chrono>

#define TSCNN_CUDA
#include "tscnn.h"

std::minstd_rand rng_engine;

template<typename T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T rng(T v) {
	std::uniform_int_distribution<T> dis(0, v - 1);
	return dis(rng_engine);
}
template<typename T, typename std::enable_if<std::is_floating_point<T>::value>::type* = nullptr>
T rng(T v) {
	std::uniform_real_distribution<T> dis(0, v);
	return dis(rng_engine);
}


using namespace tscnn;

nn<float> make_feedforward_network(size_t inputs, size_t outputs, size_t hidden_size, size_t hidden_layers) {
	nn<float> r;

	auto in = r.make_input(inputs);

	unit_ref h = in;
	for (size_t i = 0; i < hidden_layers; ++i) {
		h = r.make_tanh(r.make_linear(hidden_size, h));
		//h = r.make_linear(hidden_size, h);
	}

	r.make_output(r.make_linear(outputs, h));
	//r.make_output(r.make_sigmoid(r.make_linear(outputs, h)));

	return r;
}


int main() {

	try {

		auto a = make_feedforward_network(389, 1, 4096, 3);
		auto output_gradient_ref = a.new_gradient(a.outputs[0].gradients_index);
		a.construct();

		std::vector<float> weights(a.total_weights);
		for (auto& v : weights) {
			v = -0.1f + rng(0.2f);
		}

		std::vector<float> grad(a.total_weights);

		auto cuda_grad = a.new_vector_ref(a.total_weights);
		auto cuda_target = a.new_vector_ref(1);
		auto cuda_loss = a.new_vector_ref(1);

		int batch_size = 28;
		a.set_batch_size(batch_size);

		auto cuda_weights = a.new_vector_ref(a.total_weights);

		criterion_mse<float> criterion;
		a.make_cuda_kernels(criterion);

		a.allocate_cuda_values();

		printf("a.values.size() is %d\n", a.values.size());

		float* input = a.get_values(a.inputs[0].output);
		float* output = a.get_values(a.outputs[0].output);
		float* output_gradient = a.get_values(output_gradient_ref);

		for (int i = 0; i != a.inputs[0].output.size; ++i) input[i] = 0;

		input[0] = 0.5f;
		//input[1] = 2;
		//input[2] = 3;
		//input[3] = 4;

		auto start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i != 100; ++i) a.forward(a, weights.data());
		auto t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1000>>>(std::chrono::high_resolution_clock::now() - start).count();
		printf("forward took %gms\n", t);

		printf("cpu output: %g\n", output[0]);

		float target = 1.0f;
		float loss = 0.0f;

		//output_gradient[0] = -1.0f;
		criterion.forward(1, output, &target, &loss);

		printf("loss %g\n", loss);

		criterion.backward(1, output, &target, output_gradient);

		printf("output_gradient %g\n", output_gradient[0]);

		a.backward(a, weights.data(), grad.data());

		for (auto& v : grad) {
			//printf("grad %g\n", v);
		}

		printf("sum of grads: %g\n", std::accumulate(grad.begin(), grad.end(), 0.0f));

		output[0] = -1.0f;

		a.copy_to_cuda(weights.data(), cuda_weights, 0);
		for (int i = 0; i != batch_size; ++i) {
			input[0] = i + 0.5f;
			a.copy_to_cuda(input, a.inputs[0].output, i);
		}

		start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i != 100; ++i) {
			a.cuda_forward(a, cuda_weights, batch_size);
			a.cuda_synchronize();
		}

		t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1000>>>(std::chrono::high_resolution_clock::now() - start).count();
		printf("cuda forward took %gms\n", t);

		for (int i = 0; i != batch_size; ++i) {
			a.copy_to_cpu(a.outputs[0].output, output, i);

			printf("cuda output: %g\n", output[0]);
			output[0]++;
		}

		for (int i = 0; i != batch_size; ++i) {
			output_gradient[0] = -1.0f;
			a.copy_to_cuda(output_gradient, output_gradient_ref, i);
			a.copy_to_cuda(&target, cuda_target, 0);
			a.cuda_memset(cuda_grad, 0, i);
		}

		start = std::chrono::high_resolution_clock::now();
		a.cuda_forward_backward(a, cuda_weights, cuda_grad, a.outputs[0].output, 1, cuda_target, cuda_loss, output_gradient_ref, batch_size);
		a.cuda_synchronize();
		t = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1, 1000>>>(std::chrono::high_resolution_clock::now() - start).count();
		printf("cuda forward_backward took %gms\n", t);

		for (int i = 0; i != batch_size; ++i) {
			a.copy_to_cpu(cuda_loss, &loss, i);
			a.copy_to_cpu(output_gradient_ref, output_gradient, i);
			a.copy_to_cpu(a.outputs[0].output, output, i);

			printf("cuda loss: %g  gradient: %g\n", loss, output_gradient[0]);

			printf("cuda output: %g\n", output[0]);
			output[0]++;
		}

		for (int i = 0; i != batch_size; ++i) {
			a.copy_to_cpu(cuda_grad, grad.data(), i);
			for (auto& v : grad) {
				//printf("grad %g\n", v);
			}
			printf("sum of grads: %g\n", std::accumulate(grad.begin(), grad.end(), 0.0f));
		}

	} catch (const std::exception& e) {
		printf("exception %s\n", e.what());
	}

	return 0;
}

