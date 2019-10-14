container('milvus-build-env') {
    timeout(time: 5, unit: 'MINUTES') {
        dir ("milvus_engine") {
            dir ("core") {
                gitlabCommitStatus(name: 'Packaged Engine') {
                    if (fileExists('milvus')) {
                        try {
                            sh "tar -zcvf ./${PROJECT_NAME}-engine-${PACKAGE_VERSION}.tar.gz ./milvus"
                            def fileTransfer = load "${env.WORKSPACE}/ci/function/file_transfer.groovy"
                            fileTransfer.FileTransfer("${PROJECT_NAME}-engine-${PACKAGE_VERSION}.tar.gz", "${PROJECT_NAME}/engine/${JOB_NAME}-${BUILD_ID}", 'nas storage')
                            if (currentBuild.resultIsBetterOrEqualTo('SUCCESS')) {
                                echo "Download Milvus Engine Binary Viewer \"http://192.168.1.126:8080/${PROJECT_NAME}/engine/${JOB_NAME}-${BUILD_ID}/${PROJECT_NAME}-engine-${PACKAGE_VERSION}.tar.gz\""
                            }
                        } catch (exc) {
                            updateGitlabCommitStatus name: 'Packaged Engine', state: 'failed'
                            throw exc
                        }
                    } else {
                        updateGitlabCommitStatus name: 'Packaged Engine', state: 'failed'
                        error("Milvus binary directory don't exists!")
                    }
                }
            }
        }
    }
}
