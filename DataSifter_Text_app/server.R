library(shiny)
library(reticulate)

VIRTUALENV_NAME = "env_sifter"
PYTHON_DEPENDENCIES = c('numpy', 'autocorrect', 'boto3', 'botocore','certifi','chardet','future','gensim'
                        , 'idna', 'itsdangerous', 'jmespath', 'joblib', 'MarkupSafe', 'nltk', 'pandas', 
                        'python-dateutil', 'pytorch-pretrained-bert', 'pytz', 'rake-nltk', 'regex', 'scikit-learn',
                        'scipy', 'setuptools', 'six', 'scikit-learn', 'scipy', 'sklearn', 'smart-open'
                        , 'threadpoolctl', 'torch')

Sys.setenv(PYTHON_PATH = 'python3')
Sys.setenv(VIRTUALENV_NAME = VIRTUALENV_NAME) # Installs into default shiny virtualenvs dir
Sys.setenv(RETICULATE_PYTHON = paste0('/home/shiny/.virtualenvs/', VIRTUALENV_NAME, '/bin/python'))


virtualenv_dir = Sys.getenv('VIRTUALENV_NAME')
python_path = Sys.getenv('PYTHON_PATH')

virtualenv_create(envname = virtualenv_dir, python = python_path)
virtualenv_install(virtualenv_dir, packages = PYTHON_DEPENDENCIES, ignore_installed=TRUE)
use_virtualenv(virtualenv_dir, required = T)

# Import python functions to R
source_python('total.py')


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