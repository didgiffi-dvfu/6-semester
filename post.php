<?php
$postId = isset($_GET['id']) ? (int) $_GET['id'] : 0;

if ($postId > 0) {
    header('Location: /posts.php?id=' . $postId);
    exit();
}

header('Location: /index.php');
exit();
?>
