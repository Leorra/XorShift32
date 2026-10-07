/*
====================================================================
[+] Lock-free thread-safe XorShift32 bit random number generator [+]
[+] C++ 17 Code Standard                                         [+]
[+] https://github.com/Leorra/                                   [+]
====================================================================
*/

#pragma once

#include <atomic>
#include <cstdint>
#include <limits>

/**
 * @brief Lock-free, thread-safe Marsaglia XorShift32 pseudo-random generator.
 *
 * Properties:
 *  - Shift triple (13, 17, 5): full period of 2^32 - 1 over the non-zero states.
 *  - Output domain is [1, 2^32 - 1]; the value 0 is never produced.
 *  - Satisfies UniformRandomBitGenerator (usable with <random> distributions
 *    and std::shuffle, passed by reference).
 *  - Concurrent calls are linearizable: every successful CAS advances the
 *    shared state by exactly one step, so no step is lost or duplicated.
 *    The interleaving of outputs between threads is, however, non-deterministic.
 *
 * Limitations:
 *  - NOT cryptographically secure.
 *  - Low statistical quality compared to modern generators (e.g. PCG, xoshiro).
 *  - A single shared atomic state serializes contending threads; for hot
 *    loops prefer one instance per thread (e.g. thread_local).
 *  - Non-copyable and non-movable (inherited from std::atomic).
 */
class XorShift32 {
private:
	static constexpr std::uint32_t kDefaultSeed_ { 1337U };
	std::atomic<std::uint32_t> rng_state_ { kDefaultSeed_ };

	// Guarantees the atomic never falls back to an internal lock.
	static_assert(std::atomic<std::uint32_t>::is_always_lock_free);

public:
	// ---------------------------------------------------------------
	// UniformRandomBitGenerator interface requirements
	// ---------------------------------------------------------------
	using result_type = std::uint32_t;

	// Smallest value the generator can return (state 0 is unreachable).
	[[nodiscard]] static constexpr result_type min() noexcept {
		return 1U;
	}

	// Largest value the generator can return.
	[[nodiscard]] static constexpr result_type max() noexcept {
		return std::numeric_limits<result_type>::max();
	}

	[[nodiscard]] result_type operator()() noexcept { return xorShift32(); }

	// ---------------------------------------------------------------
	// Construction and seeding
	// ---------------------------------------------------------------
	explicit XorShift32(const std::uint32_t seed = kDefaultSeed_) noexcept { setSeed(seed); }

	XorShift32(const XorShift32&)            = delete;
	XorShift32& operator=(const XorShift32&) = delete;

	// Replaces the state. Seed 0 is the sole absorbing state of XorShift
	// and is therefore mapped to the default seed.
	// Note: small or low-entropy seeds yield weakly diffused initial outputs;
	// consider scrambling the seed (e.g. SplitMix32) or discarding a few values.
	void setSeed(const std::uint32_t seed) noexcept {
		rng_state_.store(seed == 0U ? kDefaultSeed_ : seed, std::memory_order_relaxed);
	}

	// ---------------------------------------------------------------
	// Core generator
	// ---------------------------------------------------------------

	// Advances the shared state by one XorShift32 step and returns the new
	// state, uniformly distributed over [1, 2^32 - 1].
	// Relaxed ordering suffices: the state is the only shared datum and no
	// other memory is published through it.
	[[nodiscard]] std::uint32_t xorShift32() noexcept {
		std::uint32_t current = rng_state_.load(std::memory_order_relaxed);
		std::uint32_t next    = 0U;

		do {
			// On CAS failure, 'current' is refreshed with the latest state
			// and the step is recomputed from it.
			std::uint32_t x = current;
			x ^= x << 13;
			x ^= x >> 17;
			x ^= x << 5;
			next = x;
		} while (!rng_state_.compare_exchange_weak(
			current, next, std::memory_order_relaxed, std::memory_order_relaxed));

		return next;
	}

	// ---------------------------------------------------------------
	// Derived distributions
	// ---------------------------------------------------------------

	// Uniform float in [0.0, 1.0).
	// The upper 24 bits (the float mantissa width) are scaled by 2^-24; the
	// product is exact, and the maximum 0xFFFFFF * 2^-24 stays strictly below 1.0f.
	[[nodiscard]] float getRandomFloat() noexcept {
		constexpr float kInv2p24 = 1.0F / 16777216.0F;
		return static_cast<float>(xorShift32() >> 8) * kInv2p24;
	}

	// Integer in [0, n) via a plain multiply-shift (no rejection).
	// Fast but slightly biased: 2^32 - 1 source values cannot be split evenly
	// into n buckets, giving a relative bias on the order of n / 2^32.
	// Unsuitable for strict statistical work. Returns 0 for n <= 1.
	[[nodiscard]] std::uint32_t getRandomInt(const std::uint32_t n) noexcept {
		if (n <= 1U) { return 0U; }
		// (output - 1) lies in [0, 2^32 - 2], so the product shifted by 32
		// is bounded by n - 1.
		const std::uint64_t multi = static_cast<std::uint64_t>(xorShift32() - 1U) * n;
		return static_cast<std::uint32_t>(multi >> 32);
	}

	// Unbiased integer in [0, n).
	// Lemire's nearly-divisionless method adapted to a source range of
	// R = 2^32 - 1 equally likely values x in [0, R):
	//   m = x * n,  result = floor(m / R),  low = m mod R.
	// Each result k receives floor(R / n) or floor(R / n) + 1 source values;
	// the surplus ones are exactly those with low < (R mod n) and are rejected.
	// Since (R mod n) < n, the fast-path test (low < n) avoids computing the
	// modulus in almost all cases, and division by the constant R is compiled
	// to multiplication. Returns 0 for n <= 1.
	[[nodiscard]] std::uint32_t getRandomIntUnbiased(const std::uint32_t n) noexcept {
		if (n <= 1U) { return 0U; }
		constexpr std::uint64_t R = 0xFFFFFFFFULL; // 2^32 - 1

		std::uint64_t x      = static_cast<std::uint64_t>(xorShift32()) - 1ULL;
		std::uint64_t m      = x * n;
		std::uint32_t result = static_cast<std::uint32_t>(m / R);
		std::uint32_t low    = static_cast<std::uint32_t>(m % R);

		if (low < n) {
			const std::uint32_t threshold = static_cast<std::uint32_t>(R % n);
			while (low < threshold) {
				x      = static_cast<std::uint64_t>(xorShift32()) - 1ULL;
				m      = x * n;
				result = static_cast<std::uint32_t>(m / R);
				low    = static_cast<std::uint32_t>(m % R);
			}
		}
		return result;
	}
};