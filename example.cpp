#include <cstdint>
#include <cstdio>

#include <array>
#include <vector>
#include <random>
#include <thread>
#include <chrono>

#include "tscnn.h"
using namespace tscnn;

std::default_random_engine rng_engine([] {
	std::array<unsigned int, 4> arr;
	arr[0] = std::random_device()();
	arr[1] = (unsigned int)std::chrono::high_resolution_clock::now().time_since_epoch().count();
	//arr[1] = (unsigned int)0;
	arr[2] = (unsigned int)std::hash<std::thread::id>()(std::this_thread::get_id());
	arr[3] = (unsigned int)0;
	std::seed_seq seq(arr.begin(), arr.end());
	std::default_random_engine e(seq);
	return e;
}());


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

auto make_feedforward_network(size_t inputs, size_t outputs, size_t hidden_size, size_t hidden_layers) {
	nn<> r;

	auto in = r.make_input(inputs);

	unit_ref h = in;
	for (size_t i = 0; i < hidden_layers; ++i) {
		h = r.make_sigmoid(r.make_linear(hidden_size, h));
	}

	auto out = r.make_output(r.make_linear(outputs, h));

	return r;
}


template<typename eval_F>
auto train(nn<>& network, size_t batch_size, const eval_F& eval) {

	auto output_gradient_ref = network.new_gradient(network.outputs[0].gradients_index);

	network.construct();

	std::vector<float> weights(network.total_weights);
	for (auto& v : weights) {
		v = -0.1f + rng(0.2f);
	}

	std::vector<float> grad(network.total_weights);

	criterion_mse<> criterion;

	rmsprop<> opt;
	opt.alpha = 0.9f;
	opt.learning_rate = 1e-3f;

	std::vector<float> target(network.outputs[0].output.size);

	float* input = network.get_values(network.inputs[0].output);
	float* output = network.get_values(network.outputs[0].output);
	float* output_gradient = network.get_values(output_gradient_ref);

	for (size_t i = 0; i < 100000; ++i) {

		for (auto& v : grad) v = 0.0;

		float loss = 0.0;

		for (size_t ib = 0; ib < batch_size; ++ib) {

			eval(input, target.data());

			network.forward(network, weights.data());

			float this_loss;

			criterion.forward(target.size(), output, target.data(), &this_loss);

			loss += this_loss;

			criterion.backward(target.size(), output, target.data(), output_gradient);

			network.backward(network, weights.data(), grad.data());
		}

		loss /= batch_size;

		printf("loss %g\n", loss);

		opt(weights.data(), grad.data(), grad.size());

		if (loss <= 1e-4) break;

	}

	return weights;
}

void show(nn<>& network, std::vector<float>& weights, std::vector<float> in, std::vector<float> target) {

	float* input = network.get_values(network.inputs[0].output);
	float* output = network.get_values(network.outputs[0].output);

	printf("input:");
	for (size_t i = 0; i < in.size(); ++i) {
		printf(" %g", in[i]);
		input[i] = in[i];
	}
	printf("\n");

	network.forward(network, weights.data());

	printf("output:");
	for (size_t i = 0; i < network.outputs[0].output.size; ++i) {
		printf(" %g", output[i]);
	}
	printf("\n");

	printf("errors:");
	for (size_t i = 0; i < network.outputs[0].output.size; ++i) {
		printf(" %g", target[i] - output[i]);
	}
	printf("\n");
	//printf("\n");
	
}

int main() {

	auto xor_network = make_feedforward_network(2, 1, 2, 1);
	
	auto xor_weights = train(xor_network, 100, [](float* input, float* target_output) {

		bool a = rng(2) == 0;
		bool b = rng(2) == 0;

		input[0] = a ? 1.0f : 0.0f;
		input[1] = b ? 1.0f : 0.0f;

		target_output[0] = a^b ? 1.0f : 0.0f;

	});

	auto sin_cos_network = make_feedforward_network(1, 2, 6, 2);

	auto sin_cos_weights = train(sin_cos_network, 200, [](float* input, float* target_output) {

		input[0] = rng(3.14f);

		target_output[0] = std::sin(input[0]);
		target_output[1] = std::cos(input[0]);

	});

	printf("\nxor\n--\n");
	show(xor_network, xor_weights, { 0, 0 }, { 0 });
	show(xor_network, xor_weights, { 0, 1 }, { 1 });
	show(xor_network, xor_weights, { 1, 0 }, { 1 });
	show(xor_network, xor_weights, { 1, 1 }, { 0 });

	printf("\nsin cos\n--\n");
	for (float v = 0.0; v < 3.14f; v += 3.14f / 8) {
		show(sin_cos_network, sin_cos_weights, { v }, { std::sin(v), std::cos(v) });
	}


	return 0;
}
