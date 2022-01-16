
proc optmodel;
var q1 >= 25000;
var q2 >= 25000;
var q3 >= 25000;
var q4 >= 25000;

var s1 >= 0;
var s2 >= 0;
var s3 >= 0;
var s4 >= 0;

con 62500 + 12000 - s1 >= 30000 ;
con 62500 + 12000 - s1 + 18000 - s2 >= 30000;
con 62500 + 12000 - s1 + 18000 - s2  + 20000 - s3 >= 30000;
con 62500 + 12000 - s1 + 18000 - s2  + 20000 - s3 + 22000 - s4 >= 30000 ;


con 3*s1 >= q1;
con 3*s2 >= q2;
con 3*s3 >= q3;
con 3*s4 >= q4;

con q1 + s1 = 55175;
con q2 + s2 = 56676;
con q3 + s3 = 60870;
con q4 + s4 = 57638;



min total_cost = (q1 + q2 + q3 + q4)*0.15 + 0.18*(s1 + s2) + 0.10*(s3 + s4);

solve;
print total_cost;
print q1 q2 q3 q4;
print s1 s2 s3 s4;

quit;






proc optmodel;
var q1 >= 35000;
var q2 >= 35000;
var q3 >= 35000;
var q4 >= 35000;

var s1 >= 0;
var s2 >= 0;
var s3 >= 0;
var s4 >= 0;

con 62500 + 12000 - s1 >= 30000 ;
con 62500 + 12000 - s1 + 18000 - s2 >= 30000;
con 62500 + 12000 - s1 + 18000 - s2  + 20000 - s3 >= 30000;
con 62500 + 12000 - s1 + 18000 - s2  + 20000 - s3 + 22000 - s4 >= 30000 ;




con 3*s1 >= q1;
con 3*s2 >= q2;
con 3*s3 >= q3;
con 3*s4 >= q4;

con q1 + s1 = 55175;
con q2 + s2 = 56676;
con q3 + s3 = 60870;
con q4 + s4 = 57638;


min total_cost = (q1 + q2 + q3 + q4)*0.12 + 0.18*(s1 + s2) + 0.10*(s3 + s4);

solve;
print total_cost;
print q1 q2 q3 q4;
print s1 s2 s3 s4;

quit;
