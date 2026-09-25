// ====================================================================
// [+] Lock-free thread-safe XorShift32 bit random number generator [+] 
// https://github.com/Leorra/
// ====================================================================
#pragma once

#include <atomic>
#include <cstdint>
#include <limits>

class XorShift32 {
private:
	static constexpr std::uint32_t kDefaultSeed_ { 1337U };
	std::atomic<std::uint32_t> rng_state_ { kDefaultSeed_ };

public:
	// URBG Interface Requirements
	using result_type = std::uint32_t;

	[[nodiscard]] static constexpr result_type min() noexcept {
		return std::numeric_limits<result_type>::min();
	}

	[[nodiscard]] static constexpr result_type max() noexcept {
		return std::numeric_limits<result_type>::max();
	}

	result_type operator()() noexcept {
		return xorShift32();
	}

	explicit XorShift32(std::uint32_t seed = kDefaultSeed_) noexcept { setSeed(seed); }

	void setSeed(std::uint32_t seed) noexcept {
		rng_state_.store(seed == 0U ? kDefaultSeed_ : seed, std::memory_order_relaxed);
	}

	// Lock-free thread-safe XorShift32 bit random number generator
	[[nodiscard]] std::uint32_t xorShift32() noexcept {
		std::uint32_t current = rng_state_.load(std::memory_order_relaxed);
		std::uint32_t next = 0;

		do { std::uint32_t x = current; x ^= x << 13; x ^= x >> 17; x ^= x << 5; next = x; } while (!rng_state_.compare_exchange_weak(
			current, next, std::memory_order_relaxed, std::memory_order_relaxed
		)); return next;
	}

	// Floating-point random number generator in the range [0.0, 1.0)
	[[nodiscard]] float getRandomFloat() noexcept {
		static constexpr float kInvMax = 1.0f / 16777216.0f;
		return static_cast<float>(xorShift32() >> 8) * kInvMax;
	}

	// Fast biased variant for hot loops
	[[nodiscard]] std::uint32_t getRandomInt(std::uint32_t n) noexcept {
		if (n <= 1) [[unlikely]] { return 0; }
		const std::uint64_t multi = static_cast<std::uint64_t>(xorShift32()) * n;
		return static_cast<std::uint32_t>(multi >> 32);
	}

	// Unbiased integer random number generator in the range [0, n)
	// Lemire's reduction with rejection sampling for non-power-of-two bounds
	[[nodiscard]] std::uint32_t getRandomIntUnbiased(std::uint32_t n) noexcept {
		if (n <= 1) [[unlikely]] { return 0; }
		std::uint64_t multi = static_cast<std::uint64_t>(xorShift32()) * n;
		std::uint32_t low = static_cast<std::uint32_t>(multi);
		if (low < n) {
			const std::uint32_t threshold = (0U - n) % n;
			while (low < threshold) {
				multi = static_cast<std::uint64_t>(xorShift32()) * n;
				low = static_cast<std::uint32_t>(multi);
			}
		} return static_cast<std::uint32_t>(multi >> 32);
	}
};