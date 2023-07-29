# move tests directory, which causes error
mv correlations/tests correlations/.tests;

# build docs
pdoc --html ./ --force;

# move tests directory back
mv correlations/.tests correlations/tests;
