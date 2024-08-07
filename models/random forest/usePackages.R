loadPkg <- function(pkgname){
    # Test to see if package pkgname is installed. 
    if(require(pkgname,character.only = TRUE)){
        # paste0() concatenates strings without any separator
        print(paste0("'",pkgname,"' is loaded correctly"))
    } else {
        # The require() function returned FALSE so we will try to install the package from the CRAN site
        print(paste0("Trying to install '",pkgname,"'"))
        install.packages(pkgname,character.only = TRUE,repos="http://cran.us.r-project.org")
        if(require(pkgname,character.only = TRUE)){
            print(paste0("'",pkgname,"' is installed and loaded."))
        } else {
            print(paste0("Could not install '",pkgname,"'"))
        }
    }
}

# If we provide a vector of package names, we can load them all as follows:
loadPkgs <- function(pkgnames){
    for (pkgname in pkgnames)loadPkg(pkgname)
}
