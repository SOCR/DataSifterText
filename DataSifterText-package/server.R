library(shiny)
library(reticulate)

# FIXME: change it to your local virtual environment
reticulate::use_virtualenv("~/env")
reticulate::use_python('env/bin/python3')

# Import python functions to R
# FIXME: change it to your local path
source_python('/home/yitongli/DataSifterText/DataSifterText-package/total.py')


# Define server logic required to generate and plot a random distribution
shinyServer(function(input, output, session) {
  
  # Expression that generates a plot of the distribution. The expression
  # is wrapped in a call to renderPlot to indicate that:
  #
  #  1) It is "reactive" and therefore should be automatically 
  #     re-executed when inputs change
  #  2) Its output type is a plot 
  #

  observeEvent(input$goTable, {
    req(input$file1)
    #pd <- import("pandas")
    input <- read.csv(input$file1$datapath)
    # input <- input$file1$datapath
    output$table <- renderDataTable({
      compute(input)
    })
  })
  
})