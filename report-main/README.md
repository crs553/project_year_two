# Report

Contains the main report for the INT2 project, as well as the original IEEE template (for reference).

## Prerequisites

Please ensure that a complete LaTeX distribution is installed.

### Windows, macOS
Install `MiKTeX` using the net installer to ensure that all components are selected and installed.

### Ubuntu
Install the `texlive-full` package.

##  Building

Use this command to compile the report. (It may take a few moments on the first run.)

```
pdflatex report.tex
```

The output PDF will be called `report.pdf`, in the same folder. Please note that you may also have to run `biber report` in order to ensure the bibliography prints correctly.

## Contributing

Images should be placed in the `images` folder. Check `.gitignore` to see which files will be excluded from the repo.
