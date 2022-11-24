FROM underworldcode/underworld:2.12.2b

# This command will copy in all the files in your repo.
COPY --chown=jovyan:users . /home/jovyan/community_model

# Set working directory to where we've put files. 
WORKDIR /home/jovyan/community_model

# Create symbolic link to documentation. 
RUN ln -s .. underworld_documentation
