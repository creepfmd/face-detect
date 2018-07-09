# face-detect
Finds all faces in video stream and groups each face on frame in separate folder.

On next frame compares found faces with existing in folders. If found again - stacks in folder, if not - stacks in new folder.

And so on.

Database of images for each face is limited with 45.

The more similar faces with small differences found, the smarter becomes recognition.
