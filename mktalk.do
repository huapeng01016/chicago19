/* 
	Author: 	Hua Peng
	Date:		Jun 17, 2019
	Purpose:	Build reveal.js slides deck for 2019 Chicago	
*/

dynpandoc rep.md, 	/// 
	sav(index.html)	///
	replace 		/// 
	to(revealjs) 	/// 
	pargs(-s --template=revealjs.html  	/// 
		--self-contained	/// 
		--section-divs	/// 
		--variable theme="stata"	///
		  )

exit

