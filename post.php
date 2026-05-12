<?php
$postId =  $_GET['id'];

if ($postId > 0) {
    header('Location: /posts.php?id=' . $postId);
    exit();
}

header('Location: /index.php');
exit();
?>
