sysuse auto, clear

generate fuel = 100/mpg

label variable fuel "Fuel consumption (Gallons per 100 Miles)"

scatter fuel weight, mcolor(blue%50)

regress fuel weight

mata:

// get data
st_view(Y=.,.,("fuel"), .)

st_view(X=.,.,("weight"), .)

// expand X to include constant term
X=X,J(rows(X),1,1)

// calculate b
b=invsym(X'*X)*X'*Y

// residual
e = Y- X*b

// variance-covariance
v=(e'*e)/(rows(X)-cols(X))*invsym(X'*X) 

// standard errors
se=sqrt(diagonal(v))

// t-statistic - beta divided by the standard error for each element
t=b:/se

//  p-values 
p=2*ttail(rows(X)-cols(X),abs(t))

reg_res = b,se,t,p

reg_res
end
