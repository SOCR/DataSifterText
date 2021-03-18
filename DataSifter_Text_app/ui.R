library(shiny)

# Define UI for application that plots random distributions 
shinyUI(fluidPage(
  
  # Application title
  headerPanel("DataSifter Text!"),
  sidebarLayout(
    sidebarPanel(
      # upload a file
      fileInput("file1", "Choose CSV File",
                multiple = FALSE,
                accept = c("text/csv",
                           "text/comma-separated-values,text/plain",
                           ".csv")),
      # horizontal line ----
      tags$hr(),
      
      selectInput("method", "Obfuscation methods:",
                  c("keyword", "position")),
      
      actionButton('goTable', 'Compute!'),
    ),
    
    mainPanel(
      dataTableOutput('table') 
    )
  ),
  
  # Sidebar with a slider input for number of observations
  
  # Show a plot of the generated distribution

))