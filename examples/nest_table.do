putdocx clear 
putdocx begin, landscape

sysuse auto, clear

generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"

putdocx paragraph, style("Heading1")
putdocx text ("Produce a table with tables nested in cells")

	// table may be created in memory using -memtable- option
regress fuel weight if foreign, cformat(%9.4f) 
putdocx table tbl_f = etable, memtable
regress fuel weight if !foreign, cformat(%9.4f) 
putdocx table tbl_d = etable, memtable
	// add tables in memory into cells of another tbale
putdocx table tbl_l = (2, 2)
putdocx table tbl_l(1, 1)  = ("Foreign"), halign(center)
putdocx table tbl_l(1, 2)  = ("Domestic"), halign(center)
putdocx table tbl_l(2, 1)  = table(tbl_f)
putdocx table tbl_l(2, 2)  = table(tbl_d)	

putdocx save nest_table.docx, replace

exit
