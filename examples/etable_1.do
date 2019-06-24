putdocx clear 
putdocx begin

sysuse auto, clear

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading1")
putdocx text ("Produce a table from regression results")

regress fuel weight

putdocx table tbl_est = etable

putdocx save etable_1.docx, replace

exit
