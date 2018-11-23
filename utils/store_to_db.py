#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 23:23:53 2018

@author: malrawi
"""

# from datetime import datetime
from sqlalchemy import Column, Integer, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


Base = declarative_base()

class model_output(Base):
    #Tell SQLAlchemy what the table name is and if there's any table-specific arguments it should know about
    __tablename__ = 'model_output'
    __table_args__ = {'sqlite_autoincrement': True}
    #tell SQLAlchemy the name of column and its attributes:
    id = Column(Integer, primary_key=True, nullable=False)     
    epoch = Column(Integer)    
    iteration = Column(Integer)
    loss = Column(Float)    
   
db_path = '/home/malrawi/Desktop/'
db_file_name = 'rawi_db_test.db'

#Create the database
engine = create_engine('sqlite:////'+ db_path + db_file_name)
Base.metadata.create_all(engine)

#Create the session
session = sessionmaker()
session.configure(bind=engine)
s = session()

try:
    
    data = [[11,22,33], [44,55,66]]

    for i in data:
        record = model_output(loss = i[0], 
                              epoch = i[1],
                              iteration = i[2])
        
        
        s.add(record) # Add record
        
    s.add(model_output(loss=22))
    s.commit() #Attempt to commit all the records
except:
    s.rollback() #Rollback the changes on error
finally:
    s.close() #Close the connection
   
