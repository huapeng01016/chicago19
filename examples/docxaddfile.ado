*! version 1.0.0  07jun2018
program define docxaddfile
	version 15.0
	
	syntax anything [, stopat(string) append]
	
	gettoken file opargs : anything
	local srcfile = strtrim("`file'")
	confirm file "`srcfile'"

	if ("`append'" == "") {	
		putdocx paragraph, font("courier new", 9)
	}
	mata: __docx_addtextfile(`"`srcfile'"', `"`stopat'"')
end


mata:

void  __docx_addtextfile(string scalar filename, string scalar stopat)
{
	real scalar fh
	string scalar line 
	string scalar cmd
	real scalar docx_id
	
	docx_id = strtoreal(st_global("ST__DOCX_ID"))
	if(docx_id < 0) {
errprintf("invalid docx id\n")
exit(198)
	}
	
	fh = fopen(filename, "r")
	while ((line=fget(fh))!=J(0,0,"")) {
		if(ustrpos(line, stopat) > 0) {
			break 
		}

		(void)_docx_paragraph_add_text(docx_id, line)
		(void)_docx_text_add_cr(docx_id)
	}
	fclose(fh)
}
end
