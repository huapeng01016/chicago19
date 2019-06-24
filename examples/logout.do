putdocx clear 
putdocx begin

putdocx paragraph, style("Heading1")
putdocx text ("Include output from Stata commands")

log using logout.log, text replace nomsg
sysuse auto, clear
generate fuel = 100/mpg
label variable fuel "Fuel consumption (Gallons per 100 Miles)"
describe fuel weight
regress fuel weight
log close

docxaddfile logout.log, stopat("log close")

cap erase logout.log

putdocx save logout.docx, replace

exit
