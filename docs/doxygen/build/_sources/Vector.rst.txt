*****************
Vector Operations
*****************

This section documents operations that apply to vector-like containers, including C-style arrays,
``std::vector``, and ``std::array``. These functions support both ``float`` and ``double`` types and
utilize SIMD acceleration where supported for improved performance.

.. note::

   These operations are type-constrained to floating point types (`float` and `double`) and assert
   that input containers have matching lengths.

Dot Product
===========

The **dot product** (also called the scalar product or inner product) is a fundamental operation
in linear algebra that takes two equal-length vectors and returns a single scalar value. It is
used in numerous applications such as projections, measuring vector similarity (cosine similarity),
and computing matrix products.

Mathematically, the dot product of two vectors :math:`\mathbf{a}` and :math:`\mathbf{b}`, each with
:math:`n` elements, is defined as:

.. math::

   \mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i

This operation multiplies each corresponding pair of elements from the input vectors and sums the results.


.. cpp:function:: dot(const T* a, const T* b, std::size_t size)

   Computes the dot product of two raw arrays of type ``T`` (``float`` or ``double``).

   :param a: Pointer to the first input array.
   :param b: Pointer to the second input array.
   :param size: Number of elements in the arrays.
   :returns: The scalar dot product ``∑ aᵢ × bᵢ``.
   :raises: Assertion failure if ``a`` and ``b`` are not the same size.

   **Example (raw array):**

   .. code-block:: cpp

      float a[] = {1.0f, 2.0f, 3.0f};
      float b[] = {4.0f, 5.0f, 6.0f};
      float result = dot(a, b, 3);  // returns 32.0f

.. cpp:function:: dot(const std::vector<T>& a, const std::vector<T>& b)

   Computes the dot product of two ``std::vector`` containers of type ``T``.

   :param a: First input vector.
   :param b: Second input vector.
   :returns: The scalar dot product ``∑ aᵢ × bᵢ``.
   :raises: Assertion failure if vectors are not the same size.

   **Example (std::vector):**

   .. code-block:: cpp

      std::vector<double> a = {1.0, 2.0, 3.0};
      std::vector<double> b = {4.0, 5.0, 6.0};
      double result = dot(a, b);  // returns 32.0

.. cpp:function:: dot(const std::array<T, N>& a, const std::array<T, N>& b)

   Computes the dot product of two ``std::array`` containers of type ``T`` and fixed size ``N``.

   :param a: First input array.
   :param b: Second input array.
   :returns: The scalar dot product ``∑ aᵢ × bᵢ``.
   :raises: No runtime size error; size mismatch is a compile-time error.

   **Example (std::array):**

   .. code-block:: cpp

      std::array<float, 3> a = {1.0f, 2.0f, 3.0f};
      std::array<float, 3> b = {4.0f, 5.0f, 6.0f};
      float result = dot(a, b);  // returns 32.0f


