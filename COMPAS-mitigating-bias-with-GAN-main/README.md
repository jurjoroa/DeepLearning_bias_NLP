# COMPAS - Mitigating Bias with GAN

This ML project focuses on **bias** in data and its effects on real life. 

The use case is given by the so-called **COMPAS** algorithm, which stands for **C**orrectional **O**ffender **M**anagement **P**rofiling for **A**lternative **S**anctions. Such an algorithm has been in use in the USA to assess the likelihood (risk) of a defendant becoming a recidivist in the next two years, by computing a
a score between 1 and 10. 

While we know that the input of COMPAS is given by the defendant’s answers to a 137 questions questionnaire, we do not have access to such questionnaire or to the actual code, which is a trade secret. Regardless, scores have been used by US judges to decide whether to release or detain a defendant before his/her trial and to inform other decisions determining parole and sentencing. 

In 2016, **ProPublca** compared COMPAS scores of 7.000 arrested people in 2013 and 2014 with their actual criminal activity within two years to assess the actual accuracy of the algorithm in predicting the risk of recidivism, following the concerns of some US courts members, trying to verify whether the algorithm is biased against African American defendants. More details can be found here: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing .

<img src="https://github.com/MiriamGiuliani/COMPAS-mitigating-bias-with-GAN/blob/302e02c2141c0da908bd74a243db3a9ba470d2fd/images/ProPublica_results_summary.png" alt="banner that saysmy name">

The first part of this project is to **replicate the results** of the COMPAS unknown algorithm, by building a simple Feed Forward Neural Network, starting from the data made available by ProPublica (link: https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv). 

Then, by using a Generative Adversarial Network, I tried to **mitigate the biased results** obtained by the first architecture, by training a generator to predict the recidivism risk of a defendant, in a competition with a discriminator trained to distinguish between black and caucasian people. By fooling the discriminator, the GAN architecture can then help in eliminating the bias in the training process. 

Finally, I **compared the results** of the two models with the COMPAS predictions and verified how a better degree of _algorithmic fairness_ can be reached by using a different approach to the problem. 

A more detailed description of the problem and models used can be found in the following file: https://github.com/MiriamGiuliani/COMPAS-mitigating-bias-with-GAN/blob/302e02c2141c0da908bd74a243db3a9ba470d2fd/results_presentation.pdf

