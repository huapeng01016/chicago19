putdocx clear 
putdocx begin

putdocx paragraph, style("Heading1")
putdocx text ("A fuel consumption study of Stata's auto dataset")

putdocx paragraph
putdocx text ("We conduct a study of the fuel consumption of cars in Stata's auto dataset.") 

sysuse auto, clear

putdocx paragraph, style("Heading2")
putdocx text ("Perform data transformation")

putdocx paragraph
putdocx text ("We generate a variable, ")
putdocx text ("fuel"),  bold
putdocx text (", in the unit of Gallons per 100 Miles ")  
putdocx text ("based on the variable ")
putdocx text ("mpg"), bold 
putdocx text (".") 

putdocx paragraph, font("Courier New", 12)
putdocx text (". generate fuel = 100/mpg"), linebreak
putdocx text (`". label variable fuel "Fuel consumption (Gallons per 100 Miles)""')

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading2")
putdocx text ("Examine the variables")

putdocx paragraph
putdocx text ("We examine variables for possible errors in the data.")

preserve
describe fuel weight, replace clear

putdocx paragraph, style("Heading3")
putdocx text ("Describe the variables")

		// generate the describe table from dataset in memory
sort name
putdocx table tbl_desc = data("name type format vallab varlab"), ///
	varnames				///
	border(start, nil) 		/// 
	border(insideV, nil) 	/// 
	border(insideH, nil) 	/// 
	border(end, nil)

putdocx table tbl_desc(1, 1) = ("Name")
putdocx table tbl_desc(1, 2) = ("Storage type")
putdocx table tbl_desc(1, 3) = ("Display format")
putdocx table tbl_desc(1, 4) = ("Value label")
putdocx table tbl_desc(1, 5) = ("Variable label")
putdocx table tbl_desc(1, .), border(bottom, single)

restore

		// add a summarize table from saved results
putdocx paragraph, style("Heading3")
putdocx text ("Summarize the variables")

tabstat fuel weight, stats(n mean sd min max) save
matrix stats = r(StatTotal)'

putdocx table tbl_summ = matrix(stats),  ///
		nformat(%9.4g)					 ///
        rownames colnames                ///
        border(start, nil)               ///
        border(insideV, nil)             ///
        border(insideH, nil)             ///
        border(end, nil)

putdocx table tbl_summ(1, 1) = ("Variable")
putdocx table tbl_summ(1, 2) = ("Observations")
putdocx table tbl_summ(1, 3) = ("Mean")
putdocx table tbl_summ(1, 4) = ("Standard Deviation")
putdocx table tbl_summ(1, 5) = ("Min")
putdocx table tbl_summ(1, 6) = ("Max")
putdocx table tbl_summ(1, .), border(bottom, single)

summarize weight
local min : display %4.2f `r(min)'
local max : display %4.2f `r(max)'
local range : display %4.2f `r(max)'-`r(min)'

putdocx paragraph
putdocx text ("The variable ")
putdocx text ("weight"), bold 
putdocx text (" has minimum value `min', maximum value `max', ")
putdocx text ("and range `range'.")

putdocx paragraph, style("Heading2")
putdocx text ("Plot fuel consumption and vehicle weight")

scatter fuel weight, mcolor(blue%50)
graph export temp.png, replace 
putdocx paragraph, halign(center)
putdocx image temp.png, width(4) linebreak
putdocx text ("Figure 1: scatter plot fuel consumption and weight"), bold

putdocx paragraph, style("Heading2")
putdocx text ("Explore relationship between fuel consumption and vehicle weight - linear regression")

regress fuel weight

		// add the regression table
putdocx table tbl_reg = etable

matrix define eb = e(b)

putdocx paragraph
local eb11 : display %6.4f eb[1,1]
if `eb11' < 0 {
putdocx text ("The regression shows that for every unit increase in weight, a ")
putdocx text ("-`eb11'"), italic 
putdocx text (" unit decrease in fuel consumption is predicted.")  
}
else {
putdocx text ("The regression shows that for every unit increase in weight, a ")
putdocx text ("`eb11'"), italic
putdocx text (" unit increase in fuel consumption is predicted.")  
}

putdocx pagebreak
putdocx paragraph, style("Heading2")
putdocx text ("Produce a table from -estimates table-")

putdocx paragraph
putdocx text ("We list the results from two regressions.")

quietly regress fuel weight gear turn
estimates store model1
quietly regress fuel weight gear turn foreign
estimates store  model2
estimates table model1 model2, 	///
	varlabel b(%7.4f) 			/// 
	stats(N r2 r2_a) star

putdocx table tbl_est = etable, width(60%) halign(center)

putdocx pagebreak
putdocx paragraph, style("Heading2")
putdocx text ("Produce a table from community-contributed -esttab-")

putdocx paragraph
putdocx text ("We list the results from the same two regressions as above using -esttab- and -putdocx-. ")
putdocx text ("-esttab- is a popular community-contributed command which generates tables for report.")

		//table from esttab
preserve
cap erase esttab_ex.csv
eststo clear
eststo, title("Fuel consumption"): quietly regress fuel weight gear turn
eststo, title("Fuel consumption"): quietly regress fuel weight gear turn foreign
esttab using esttab_ex.csv, 		///
	b(3) t(2) r2(2) ar2(2)			/// 
	plain star notes par label 		/// 
	title(Regression table using -esttab-) width(80%)

import delimited using esttab_ex.csv, clear
putdocx table d = data(_all), border(all,nil)
putdocx table d(1,.), border(bottom,double)
putdocx table d(3,.), border(bottom, dotted)
putdocx table d(1,1), colspan(3) halign(center)
putdocx table d(14,.), border(top,dotted)
putdocx table d(17,.), border(top,double)
putdocx table d(17,1), colspan(3) halign(left) italic
putdocx table d(18,1), colspan(3) halign(left) italic
restore

putdocx sectionbreak, landscape
putdocx paragraph, style("Heading2")
putdocx text ("Relationship based on car type - a nested table")

putdocx table tbl_l = (4, 2), border(all, nil) border(top, double) border(bottom, double)

tabstat fuel weight if foreign, stats(n mean sd min max) save
matrix stats = r(StatTotal)'
putdocx table tbl_summ1 = matrix(stats),  ///
		memtable						  ///	
		nformat(%9.4g)					  ///
        rownames colnames                 ///
        border(start, nil)                ///
        border(insideV, nil)              ///
        border(insideH, nil)              ///
        border(end, nil)

putdocx table tbl_summ1(1, 1) = ("")
putdocx table tbl_summ1(1, 2) = ("Obs")
putdocx table tbl_summ1(1, 3) = ("Mean")
putdocx table tbl_summ1(1, 4) = ("SD")
putdocx table tbl_summ1(1, 5) = ("Min")
putdocx table tbl_summ1(1, 6) = ("Max")
putdocx table tbl_summ1(1, .), border(bottom, dashed) valign(bottom)

tabstat fuel weight if !foreign, stats(n mean sd min max) save
matrix stats = r(StatTotal)'
putdocx table tbl_summ2 = matrix(stats),  ///
		memtable						  ///	
		nformat(%9.4g)					  ///
        rownames colnames                 ///
        border(start, nil)                ///
        border(insideV, nil)              ///
        border(insideH, nil)              ///
        border(end, nil)

putdocx table tbl_summ2(1, 1) = ("")
putdocx table tbl_summ2(1, 2) = ("Obs")
putdocx table tbl_summ2(1, 3) = ("Mean")
putdocx table tbl_summ2(1, 4) = ("SD")
putdocx table tbl_summ2(1, 5) = ("Min")
putdocx table tbl_summ2(1, 6) = ("Max")
putdocx table tbl_summ2(1, .), border(bottom, dashed) valign(bottom)
putdocx table tbl_summ2(2, 1) = ("")
putdocx table tbl_summ2(3, 1) = ("")

putdocx table tbl_l(1, 1)  = ("Foreign"), halign(center)
putdocx table tbl_l(1, 2)  = ("Domestic"), halign(center)
putdocx table tbl_l(2, 1)  = table(tbl_summ1)
putdocx table tbl_l(2, 2)  = table(tbl_summ2)

scatter fuel weight if foreign, mcolor(blue%50)
graph export temp_f.png, replace 
scatter fuel weight if !foreign, mcolor(blue%50)
graph export temp_d.png, replace 

putdocx table tbl_l(3, 1)  = image(temp_f.png)
putdocx table tbl_l(3, 2)  = image(temp_d.png)

cap erase esttab_ex.csv
eststo clear
eststo, title("Foreign") : quietly regress fuel weight gear turn if foreign
eststo, title("Domestic"): quietly regress fuel weight gear turn if !foreign
esttab using esttab_ex.csv, 		///
	b(3) t(2) r2(2) ar2(2)			/// 
	plain star par label 			///
	wide							/// 
	nomtitles

import delimited using esttab_ex.csv, clear
putdocx table d = data(_all), memtable border(all,nil)
putdocx table d(1,.), border(bottom, dashed)

putdocx table tbl_l(4, 1), colspan(2)
putdocx table tbl_l(4, 1) = table(d)

putdocx sectionbreak
putdocx paragraph, style("Heading2")
putdocx text ("Output from Stata commands")

log using outputs.log, text replace nomsg
sysuse auto, clear
generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"
regress fuel weight
mata:
st_view(Y=.,.,("fuel"), .)
st_view(X=.,.,("weight"), .)
X=X,J(rows(X),1,1)
b=invsym(X'*X)*X'*Y
v=((Y- X*b)'*(Y- X*b))/(rows(X)-cols(X))*invsym(X'*X) 
se=sqrt(diagonal(v))
t=b:/se
p=2*ttail(rows(X)-cols(X),abs(t))
b,se,t,p
end
log close

docxaddfile outputs.log, stopat("log close")

cap erase outputs.log
cap erase esttab_ex.csv
cap erase temp.png
cap erase temp_f.png
cap erase temp_d.png

putdocx save fuel_consumption.docx, replace

exit
