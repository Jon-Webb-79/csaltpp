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

Cross Product
=============

The ``cross`` function computes the cross product of two 3D vectors. It is 
for both ``float`` and ``double`` types, and 
supports C-style arrays, ``std::array``, and ``std::vector`` as input.

Given two 3-dimensional vectors:

.. math::

   \mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \\ a_3 \end{bmatrix}, \quad
   \mathbf{b} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}

Their cross product is defined as:

.. math::

   \mathbf{a} \times \mathbf{b} =
   \begin{bmatrix}
   a_2 b_3 - a_3 b_2 \\
   a_3 b_1 - a_1 b_3 \\
   a_1 b_2 - a_2 b_1
   \end{bmatrix}

The result is a vector orthogonal to both ``a`` and ``b``, with a direction 
determined by the right-hand rule and magnitude equal to the area of the 
parallelogram spanned by ``a`` and ``b``.

.. cpp:function:: template<typename T> void cross(const T a[3], const T b[3], T result[3])

Computes the cross product of two C-style arrays of length 3.

:param a: First input array (length 3)
:param b: Second input array (length 3)
:param result: Output array to hold the cross product (length 3)

Example::

   float a[3] = {1.0f, 0.0f, 0.0f};
   float b[3] = {0.0f, 1.0f, 0.0f};
   float result[3];
   slt::cross(a, b, result);
   // result = {0.0f, 0.0f, 1.0f}

.. cpp:function:: template<typename T> std::array<T, 3> cross(const std::array<T, 3>& a, const std::array<T, 3>& b)

Computes the cross product of two ``std::array<T, 3>`` inputs.

:param a: First input array
:param b: Second input array
:returns: A new array representing the cross product

Example::

   std::array<double, 3> a = {1.0, 2.0, 3.0};
   std::array<double, 3> b = {4.0, 5.0, 6.0};
   auto result = slt::cross(a, b);
   // result = {-3.0, 6.0, -3.0}

.. cpp:function:: template<typename T> std::vector<T> cross(const std::vector<T>& a, const std::vector<T>& b)

Computes the cross product of two ``std::vector<T>`` values, both of size 3.

:param a: First input vector
:param b: Second input vector
:returns: A ``std::vector<T>`` containing the cross product
:throws std::invalid_argument: If either vector is not of size 3

Example::

   std::vector<float> a = {0.0f, 0.0f, 1.0f};
   std::vector<float> b = {1.0f, 0.0f, 0.0f};
   auto result = slt::cross(a, b);
   // result = {0.0f, 1.0f, 0.0f}

