file(READ ${SRC} html)
file(WRITE ${DST} "static const char* kPlaygroundHtml = R\"STATIM_HTML(${html})STATIM_HTML\";\n")
