putdocx clear 
putdocx begin

sysuse auto, clear

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading1")
putdocx text ("Produce a table from margins")

regress mpg weight i.foreign i.rep78
margins foreign rep78 

putdocx table tbl_marg = etable

putdocx save etable_3.docx, replace

exit
