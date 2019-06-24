putdocx clear 
putdocx begin

sysuse auto, clear

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading1")
putdocx text ("Produce a table from regression results")

regress fuel weight

putdocx table tbl_est = etable
		// change cell contents
putdocx table tbl_est(3, 1) = ("Constant")
		// change the alignment of first column
putdocx table tbl_est(.,1), halign(center)	

putdocx save etable_2.docx, replace

exit
