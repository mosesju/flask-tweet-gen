from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, FileField, SubmitField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap

class GetTweetForm(FlaskForm):
    searchterm = StringField('SearchTerm', validators=[DataRequired()])
    file_name = StringField('File Name', validators=[DataRequired()])
    time = DecimalField('Time', validators=[DataRequired()])
    submit = SubmitField('submit_value')
class GenerateTweetForm(FlaskForm):
    referencephrase1 = StringField('ReferencePhrase1', validators=[DataRequired()])
    referencephrase2 = StringField('ReferencePhrase2', validators=[DataRequired()])
    referencephrase3 = StringField('ReferencePhras3', validators=[DataRequired()])
    submit = SubmitField('Submit')