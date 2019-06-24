/* 
	Author: 	Hua Peng
	Date:		Jan 17, 2018
	Purpose:	Build reveal.js slides deck for 2018 webinar	
*/

syntax [anything(name=full)]

if "`full'" != "" {
cd examples
	dyndoc fuel.txt, replace 
	dyndoc fuel_consumption.txt, replace 
	do fuel_consumption.do
	dyndoc fuel_consumption.txt todo, saving(fuel_consumption_todo.html) replace 
	dynpandoc fuel_cc.txt, saving(fuel_pandoc.html) from(markdown) /// 
     	replace
	dynpandoc fuel_cc.txt, saving(fuel_pandoc.docx) from(markdown) /// 
     	pargs("--reference-doc=reference.docx") replace
	dynpandoc fuel_cc.txt, saving(fuel_pandoc.pdf) from(markdown) replace
	do auto_78.do
	do etable_1.do
	do etable_2.do
	do etable_3.do
	do etable_4.do
	do data_table.do
	do data_table_1.do
	do nest_table.do
	do logout.do
cd ..
}

dynpandoc rep.md, 	/// 
	sav(index.html)	///
	replace 		/// 
	to(revealjs) 	/// 
	pargs(-s --template=revealjs.html  	/// 
		  --self-contained    			/// 
		  --section-divs 				/// 
		  --variable theme="stata"		///
		  )

exit

