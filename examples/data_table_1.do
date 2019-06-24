putdocx clear 
putdocx begin

sysuse auto, clear

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading1")
putdocx text ("Produce a table from saved dataset from -esttab-")

preserve
cap erase esttab_ex.csv
eststo clear
eststo, title("Model 1"): quietly regress fuel weight gear turn
eststo, title("Model 2"): quietly regress fuel weight gear turn foreign
esttab using esttab_ex.csv, 		///
	b(3) t(2) r2(2) ar2(2)			/// 
	plain star notes par label 		/// 
	title(Regression table using -esttab-) width(80%)

import delimited using esttab_ex.csv, clear
putdocx table tbl_data = data(_all)

putdocx pagebreak

			// add a table without borders
putdocx table tbl_data_1 = data(_all), border(all,nil)
			// add a double width line border at the bottom of the first row
putdocx table tbl_data_1(1,.), border(bottom,double)
			// add a dotted line border at the bottom of the third row
putdocx table tbl_data_1(3,.), border(bottom, dotted)
			// make the first cell of the first row span 3 columns
putdocx table tbl_data_1(1,1), colspan(3) halign(center)
			// add a dotted line border at the top of the 14th row 
putdocx table tbl_data_1(14,.), border(top,dotted)
			// add a double width line border at the top of the 17th row
putdocx table tbl_data_1(17,.), border(top,double)
			// make the first cells of 17th and 18th rows span 3 columns, 
			// also make the contents of the cells left aligned and italic  
putdocx table tbl_data_1(17,1), colspan(3) halign(left) italic
putdocx table tbl_data_1(18,1), colspan(3) halign(left) italic
restore

putdocx save data_table_1.docx, replace

exit
