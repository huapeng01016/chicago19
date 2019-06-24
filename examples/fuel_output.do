cscript

putdocx begin

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

log using junk.log, text replace nomsg
generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"
describe fuel weight
regress fuel weight
log close

docxaddfile junk.log, stopat("log close")
cap erase junk.log

putdocx save test.docx, replace
