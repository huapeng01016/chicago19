putdocx clear 
putdocx begin

sysuse auto, clear

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading1")
putdocx text ("Produce a table from -estimates table-")

quietly regress fuel weight gear turn
estimates store model1
quietly regress fuel weight gear turn foreign
estimates store  model2
estimates table model1 model2, 	///
	varlabel b(%7.4f) 			/// 
	stats(N r2 r2_a) star

putdocx table tbl_est = etable

putdocx save etable_4.docx, replace

exit
