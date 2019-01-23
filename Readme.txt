Description of files in ISPTWSVM codes.

-----------------Scripts------------------
1. main.m : Used to calculate sparsity for ISPTWSVM model for linear case.

2. main_kernel.m : Used to calculate sparsity for ISPTWSVM model for non-linear case.

3. main_with_optimal_c.m : Used to calculate accuracy of ISPTWSVM model for linear case.

4. main_with_optimal_c_kernel.m : Used to calculate accuracy of ISPTWSVM model for non-linear case.

5. normalize_dataset.m : Used to normalize feature values between [0, 1].

---------------Functions------------------
6. ISPTWSVM.m : Function which returns accuracy, non-zero dual variables amd training time for linear ISPTWSVM. Input taken is (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, value of c1, value of c3, epsilon value, and tau value.

7. ISPTWSVM_Kernel.m : Function which returns accuracy, non-zero dual variables amd training time for non-linear ISPTWSVM. Input taken is (in the order listed) training samples of first class, training samples of second class, testing samples, class labels of testing samples, value of c1, value of c3, value of gamma, epsilon value, and tau value.

8. RBF.m : Used to calculate RBF kernel value between two vectors u and v.