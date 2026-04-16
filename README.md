# linear-math-engine
# Linear Math Engine: Regression from Scratch

## **Project Overview**
This project demonstrates the implementation of a **Simple Linear Regression** model built entirely from scratch using Python and NumPy. Moving beyond standard library imports, this engine manually implements the optimization process through **Gradient Descent**, showcasing a deep understanding of the calculus and linear algebra that powers supervised learning.

## **Mathematical Framework**

### **1. The Hypothesis Function**
We model the relationship between variables using the linear equation:
$$\hat{y} = Xw + b$$
Where $w$ represents the weight vector (slope) and $b$ represents the bias term (intercept).

### **2. The Cost Function (MSE)**
To measure model performance, we utilize the **Mean Squared Error (MSE)** as our loss function $J$:
$$J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### **3. The Optimization: Gradient Descent**
The model minimizes the loss function by calculating the **Partial Derivatives** (Gradients) with respect to $w$ and $b$. Applying the **Chain Rule**, we derive the following gradient vectors:
* **Weight Gradient:** $\frac{\partial J}{\partial w} = \frac{-2}{n} X^T (y - \hat{y})$
* **Bias Gradient:** $\frac{\partial J}{\partial b} = \frac{-2}{n} \sum (y - \hat{y})$

### **4. The Update Rule**
In each iteration, the parameters are adjusted in the direction of the steepest descent:
$$w = w - (\alpha \cdot \frac{\partial J}{\partial w})$$
$$b = b - (\alpha \cdot \frac{\partial J}{\partial b})$$
*Where $\alpha$ represents the **Learning Rate**.*

## **Technical Implementation Highlights**
* **Object-Oriented Design:** The logic is encapsulated within a `ManualRegressor` class, following industry-standard software architecture patterns.
* **Vectorization:** Leverages NumPy's matrix operations ($X^T$) to perform high-performance computations, avoiding inefficient Python loops.
* **Convergence Visualization:** Includes Matplotlib integration to visualize the learned regression line against raw data noise.

## **Validation**
The results of this manual implementation were validated against the standard Scikit-Learn `LinearRegression` library. Both models converged to the same parameters, verifying the mathematical accuracy of the custom gradients and update rules.

## **Technical Stack**
* **Language:** Python
* **Libraries:** NumPy, Matplotlib
* **Documentation:** LaTeX
