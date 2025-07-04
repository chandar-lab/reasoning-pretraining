# Copyright (c) 2025, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env bash

# Push files to all nodes
# Usage
# syncdir.sh file [file2..]

echo Number of files to upload: $#

for file in "$@"
do
    full_path=$(realpath $file)
    parentdir="$(dirname "$full_path")"
    echo Uploading $full_path to $parentdir
    pdcp -f 1024 -R ssh -w ^/job/hosts -r $full_path $parentdir
done
