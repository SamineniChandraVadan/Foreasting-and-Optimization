
# Forecasting and Optimization
<ins> </ins>
> This repository consists of all the files of solving the problem to choose a water contract for a company, related to Sas Optimization challenge 

# Table of Contents
- [Problem Statment](#1.Problem_Statement)

- [Objective](#2.Objective)
  - [Forecasting](#2.1.Forecasting: )
   - [Dickey-Fuller Test](#Dickey-Fuller_Test:)
   - [Model parameter selection](#Model_parameter_selection:)
   - [Model selection](#Model_selection:)
  - [Optimization](#2.2.Optimization:) 
   - [Optimization problem: ](#2.2.1.Optimization_problem:)
    
- [Conclusion](#3.Conclusion:)


# Foreasting-and-Optimization
<ins/> <ins>

#1.Problem_Statement 

`Building T’s`, headquarters of the `‘XYZ corporation’`, water demands are fulfilled by two sources. The first is through `‘The Water Co.’` with which the company has a contract. The second is through their own `Water Storage Tank`.  

To meet Building T’s water requirements in the 4 weeks starting Jan 09, 2022, XYZ Corporation must choose the more cost-efficient proposal among the two proposed contracts offered by ‘The Water Co.’. The business problem is divided into 2 parts. 
  
Part A aims at forecasting water demand in the first 4 weeks using historical time series data representing the building’s water consumption. 
 
Part B of the problem aims at constructing an optimization model using water-usage forecasts from Part A and the constraints presented in the problem. 

While solving the problem, it must be ensured that the Water Storage Tank must not drop below 30,000 gallons during any week. Additionally, at least 25% of Building T’s water requirements must be met from the Water Storage Tank to secure XYZ’s membership in the Elite Environmental Corporate Sustainer Initiative. Moreover, a per-gallon treatment cost from Water Storage must be incorporated while calculating total water costs for any week.  

The 2 proposals cited by The Water Co. are as follows: - 

  15 cents ($0.15) per gallon with a minimum of 25,000 gallons purchased per week 

  12 cents ($0.12) per gallon with a minimum of 35,000 gallons purchased per week 

#2.Objective 

##2.1.Forecasting: 

The raw data set consisted of historical weekly gallons usage data of Building T over a year and a half. This usage is split into cooling and main, where ‘cooling’ is the gallons used to regulate the temperature of the building and ‘main’ is the gallons used by the employees.  

Since the trend in demand for cooling and main water usage is different, we forecasted them separately. Figure 1 shows that there is a cyclic trend in demand for cooling whereas there is no visual trend in demand for main.  

![image](https://user-images.githubusercontent.com/51246077/149674871-11c6d079-dbdc-4a23-be29-7024ece3fec3.png)

 

##Dickey-Fuller_Test:
A stationary data has the mean, variance, and autocorrelation similar across the timeline, and only a stationary data can be used for the forecast. Dickey-Fuller test helps us to understand if there is stationarity in the data. 

![image](https://user-images.githubusercontent.com/51246077/149674964-bc82a1e8-32b0-413e-ae4e-dfb14e5c3b39.png)

 
From the above results, with the p-value of < 0.05, there is enough evidence to say that the cooling data is stationary (so we use d, non-seasonality difference, as 0). Whereas in the case of employee_usage, with the p-value of > 0.05, there is not enough evidence to say that the data is stationary, so we have to apply the first difference and check for the stationarity. 

The test on the first difference data shows that the p-value is < 0.05, so we have enough evidence to say that there is stationarity in the data after the first difference. So, we use non-seasonality difference ‘d’ as 1 in the models 

##Model_parameter_selection:
Autocorrelation and Partial correlation: Autocorrelation tells us how the time series data is correlated to the previous data with a lag. And based on the correlation we can choose the lag to incorporate in the model.
                                                                       
![image](https://user-images.githubusercontent.com/51246077/149675077-82e54b8a-f0fe-4b10-8da6-0b80361b1c25.png)

![image](https://user-images.githubusercontent.com/51246077/149675110-dbae9134-a1c0-4251-957c-43c33ff60b34.png)


Figure 3: Auto and partial correlation for cooling data 

From the partial autocorrelation graph, we can say that number of lagged forecast errors to be used is 2 and from the autocorrelation, it's 1 to 6 depending on the performance of the model. Final parameters used are: Employee Usage (p,d,q) – (2,1,2), Cooling Usage (p,d,q) – (2,0,2) 

FFT analysis of the cooling and employee water usage confirms the cyclicity in the Cooling water usage data and no cyclicity is observed in the Employee usage data. 

 ![image](https://user-images.githubusercontent.com/51246077/149675130-5d4609ae-0db4-401c-9497-3fea98eeb88e.png)

Figure 4: FFT analysis of cooling (left) and Main (right) data 

  
##Model_selection: 
First 70 weeks of the data is used for initial model training and the next 22 weeks for testing. After the model selection, the model is again trained on the whole data to predict the next 4 weeks' water usage. 

 ![image](https://user-images.githubusercontent.com/51246077/149675171-04636940-f4e9-414d-a072-8f685e66f299.png)

Figure 5: RSME values of various models for cooling (left) and Main (right) data 

 

Various models including AR, MA, ARMA, ARIMA, and SARIMAX were used to model the data. SARIMAX was chosen for the final prediction of ‘main’ and ‘cooling’ based on the RMS values and the above analysis and data visualization. 

![image](https://user-images.githubusercontent.com/51246077/149675208-d4a564f8-d0dc-4402-aaa4-b7898b2f9ca8.png)
                                                                      
Table 1: RSME values of various models for cooling (left) and Main (right) data 
                                                                       
                                                                       

##2.2.Optimization: 

XYZ’s contract with The Water Co. is renewed every four weeks and they received 2 contracts for the next four weeks. 

15 cents ($0.15) per gallon with a minimum of 25,000 gallons purchased per week 

12 cents ($0.12) per gallon with a minimum of 35,000 gallons purchased per week 

The current availability of water in the Water Storage Tank is 62,500 gallons and it should drop below 30,000 gallons during any week over the next 4 weeks. The cost of treatment of water storage is $0.18 per gallon for the 1st 2 weeks and $0.10 per gallon for the next 2 weeks. Precipitation data is already estimated and table 1 shows the estimates.  

##2.2.1.Optimization_problem: 

 
![image](https://user-images.githubusercontent.com/51246077/149675317-4b54dde2-fa3a-4b5c-93a2-7c4f9d26295f.png)
                                                                       
By optimizing the equation based on the costs and constraints mentioned above while using the forecasted numbers from 2.1, we got the below results: 

 
![image](https://user-images.githubusercontent.com/51246077/149675347-e67d03ec-5816-4d63-9700-6765ab71c496.png)

 

#3.Conclusion: 
                                                                       
From the above results,we recommend XYZ corporation to go for contract 2 with The Water Co. which could save $3616. In the future, if the constraint of a minimum of 30,000 gallons to be always available in the water  storage  tank  can  be  lifted,with the  cost  of it  also  going  down, they  can  fully depend  on the storage tank and need not buy from The Water Co.                                                                      
                                                                       
                                                                       
                                                                       
